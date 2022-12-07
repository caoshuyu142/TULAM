import argparse
import logging
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset  # 导入抽象类Dataset
from tqdm import tqdm

from model import Multi_Head_Attention_LSTM
from preprocess import getData, txt2csv_foursquare, txt2csv_gowalla, coordinate2grid, checkin2trajectory, splitData
from encoder import approximateOnehotEmbed

warnings.filterwarnings("ignore")


def parse_args():
    """[This is a function used to parse command line arguments]

    Returns:
        args ([object]): [Parse parameter object to get parse object]
    """
    parse = argparse.ArgumentParser(description='TUL-Checker')
    parse.add_argument('--dataset', type=str, default="foursquare", help='dataset for experiment')
    parse.add_argument('--batchSize', type=int, default=64, help='Size of one batch')
    parse.add_argument('--epochs', type=int, default=100, help='Number of total epochs')
    parse.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parse.add_argument('--embedSize', type=int, default=512, help='Number of embedding dim')
    parse.add_argument('--numHeads', type=int, default=6, help='Number of heads')
    parse.add_argument('--threshold', type=int, default=100, help='Minimum number of recorder for one user')
    parse.add_argument('--gridSize', type=float, default=0.05, help='Size of one grid')
    parse.add_argument('--intervalHour', type=int, default=6,
                       help='Maximum interval hour of two checkin in one trajectory')
    parse.add_argument('--testNum', type=int, default=12, help='Number of trajectory in testSet for each user')
    parse.add_argument('--hiddenSize', type=int, default=400, help='Dimension of hidden layer')
    parse.add_argument('--isDense', type=str, default='S', help='Dense area or Sparse area: D/S')

    args = parse.parse_args()
    return args


def getLogger(dataset, isDense):
    """[Define logging functions]

    Args:
        dataset ([string]): [dataset name]
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)

    if isDense == 'D':
        fileHandler = logging.FileHandler(filename='./project/log/' + dataset + 'DenseArea' + '.log', mode='w')
    else:
        fileHandler = logging.FileHandler(filename='./project/log/' + dataset + 'SparseArea' + '.log', mode='w')
    fileHandler.setLevel(logging.INFO)

    consoleFormatter = logging.Formatter("%(message)s")
    fileFormatter = logging.Formatter("%(message)s")

    consoleHandler.setFormatter(consoleFormatter)
    fileHandler.setFormatter(fileFormatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    return logger


class TrData(Dataset):
    def __init__(self, data_seq):
        self.data_seq = data_seq

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return self.data_seq[idx]


def collate_fn(trs):
    onetr = trs[0]  # 取一个轨迹
    onepoint_size = onetr.size(1)  # 取轨迹中一个点的尺寸，即维度1的尺寸
    # print(onepoint_size)

    # 修正输入X的size
    input_size = onepoint_size - 1
    # print(input_size)

    # 在DataLoader中使用，拆分数据为X和Y，同时返回当前轨迹的长度。
    # data为一个batch的轨迹集
    trs.sort(key=lambda x: len(x), reverse=True)
    # 创建一个tr_lengths列表，每个值都是轨迹的真实长度（有多少位置点）
    tr_lengths = [len(sq) for sq in trs]

    ####################################################
    # 从数据集中取出时，自动填充
    trs = rnn_utils.pad_sequence(trs, batch_first=True, padding_value=0)
    #####################################################

    # print("===========")
    # print(trs)
    # print(trs.size()) #torch.Size([10, 63, 252])，因为label没有再采用onehot，而是一个整数，并且放在第一列。

    # print("====var_x=======")
    var_x = trs[:, :, 1:input_size + 1]  # 在第3个维度上取1-XXX
    # var_x = trs[:,:,1:252] #在第3个维度上取1-251
    # print(var_x)
    # print(var_x.size())
    # Var_x为torch.Size([10, 63, 251]) batchsize=10, 这个batch中最长的轨迹是63个位置点

    # print("======tmpy========")
    tmpy = trs[:, :, 0]  # 在第3个维度上取0,第0列是label

    # print(tmpy)
    # print(tmpy.size())   #torch.Size([5, 45])相当于最长的句子是45个单词
    # print("======var_y========")

    # tmpy第1维batch里的所有轨迹的label（全要），
    # 第2维是最长轨迹的位置点个数（只要第一个点对应的label即可）。
    # label采用整数后，没有第三维了。
    var_y = tmpy[:, 0]
    # print(var_y)
    # print(var_y.size())

    return var_x, var_y, tr_lengths


def train_model(model, train_data, batch_size, num_epochs, learning_rate, test_data, num_classes):
    # 采用DataLoader从数据集中获取数据（Batch方式）
    train_data_inBatch = TrData(train_data)
    # 一个Batch取5条轨迹，在取值的时候是会shuffle的。包括x和y，但在trainloader取出时会由collate_fn拆分
    train_loader = DataLoader(train_data_inBatch, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)

    # 设置优化方法和loss
    criterion = nn.CrossEntropyLoss()
    # nn.CrossEntropyLoss里面包含了softmax层，
    # 所以我们在使用nn.CrossEntropyLoss来计算loss的时候，
    # 不需要再加softmax层作为输出层，直接使用最后输出的特征向量来计算loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 增加自动调整学习率的部分##########################
    # 每次减少为原来的0.5
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

    losslist = []
    acclist = []
    # model.train()#测试时，根本就没有反向传播调整参数的代码，因此不用担心测试的时候仍然会训练
    # model.train和eval仅仅用于使用了BN和Dropout时，训练集有一个dropout rate=0.5，
    # 而测试时要求dropout rate=1，所以通过train和eval两个函数通知框架，模型是处于什么状态，而选什么参数值。
    # 至于测试时，梯度是否在计算和累加，答案是如果不在测试阶段显式加入no_grad去掉梯度。因为测试阶段不需要反向传播梯度，这个计算是没用的。
    # 但是因为测试的代码里你没有写反向传播的代码，所以这个梯度的计算不会对模型产生影响，只是费一些内存而已。
    # 所以测试阶段加不加no_grad去掉梯度都可以。

    for epoch in tqdm(range(num_epochs),desc='Training...'):
        model.train()
        ave_loss_of_epoch = 0
        count = 0
        # print("Epoch{}/{}".format(epoch, num_epochs))
        # print("Epoch {} , 学习率为{}".format(epoch,optimizer.param_groups[0]['lr']))
        # step不是train_loader的返回值，括号里才是
        for step, (X_vector, Y_vector, len_of_oneTr) in enumerate(train_loader):
            # print("step {}".format(step))
            # print("=====X_vector======")
            # print(X_vector)
            # print(X_vector.size())   #x中的第一个维度是timeBucket_Norm, 其他维度都是onehot
            # print("=====Y_vector======")
            # print(Y_vector)
            # print(Y_vector.size())
            # print("=====len_of_oneTr======")
            # print(len_of_oneTr)
            # print(len(len_of_oneTr))

            X_vector.cuda()
            Y_vector.cuda()
            # 前向传播
            out = model(X_vector, len_of_oneTr)

            # 把out从【1，5，182】压缩为[5,182]
            out = out.squeeze(0).cuda()
            # Y_vector =Y_vector.squeeze(0)
            # cross_entropy target参数只需要标签即可, 不需要传one-hot向量，且数值必须是long，不能是float
            Y_vector = Y_vector.long().cuda()
            loss = criterion(out, Y_vector)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 用于统计
            ave_loss_of_epoch = ave_loss_of_epoch + loss
            count = count + 1
            # print('Loss:{}'.format(loss))
            # if (epoch + 1) % 5 == 0: # 每 5 次epoch输出结果
            #    print('Epoch: {}, Loss: {:.5f}'.format(epoch + 1, loss))

        ####更新学习率lr，放在epoch这一层循环中。
        if optimizer.param_groups[0]['lr'] > 0.0001:
            scheduler.step()

        ave_loss_of_epoch = ave_loss_of_epoch / count
        losslist.append(ave_loss_of_epoch.item())
        # print("Average loss of Epoch{} is {}".format(epoch+1,ave_loss_of_epoch))
        # 可视化代码
        MacroP, MacroR, MacroF1, top1_correct_Rate = test_model_inTrain(model, test_data, batch_size, num_classes)

        acclist.append(top1_correct_Rate)

    return model, losslist, acclist


def test_model_inTrain(model, test_data, batch_size, num_classes):
    model.eval()
    # model.eval()#测试时，根本就没有反向传播调整参数的代码，因此不用担心测试的时候仍然会训练
    # model.train和eval这一对函数仅仅用于使用了BN和Dropout的场景下，训练集有一个dropout rate=0.5，
    # 而测试时要求dropout rate=1，所以通过train和eval两个函数通知框架，模型是处于什么状态，而选什么参数值。
    # 至于测试时，梯度是否在计算和累加，答案是如果不在测试阶段显式加入no_grad去掉梯度。因为测试阶段不需要反向传播梯度，这个计算是没用的。
    # 但是因为测试的代码里你没有写反向传播的代码，所以这个梯度的计算不会对模型产生影响，只是费一些内存而已。
    # 所以测试阶段加不加no_grad去掉梯度都可以。

    #####################计算准确率、召回率、F1-score
    # 创建多分类混淆矩阵（只针对top1的预测结果）
    ###          标签值
    #      ###类别1，类别2，类别3
    # 预  类别1
    # 测  类别2
    # 值  类别3
    pre_class = ["pre_" + str(i) for i in range(0, num_classes)]
    label_class = ["label_" + str(i) for i in range(0, num_classes)]
    CM = pd.DataFrame(index=pre_class, columns=label_class)
    CM.fillna(0, inplace=True)

    #########################计算topk准确率############
    #########计算Topk准确率
    top1 = 1

    top1_correct = 0

    total = 0
    ######################################

    # 采用DataLoader从数据集中获取数据（Batch方式）
    test_data_inBatch = TrData(test_data)
    # 一个Batch取5条轨迹，测试集不shuffle。包括x和y，但在trainloader取出时会由collate_fn拆分
    test_loader = DataLoader(test_data_inBatch, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, drop_last=True)

    # print("开始预测！")

    for step, (X_vector, Y_vector, len_of_oneTr) in enumerate(test_loader):
        # 预测
        X_vector.cuda()
        output = model(X_vector, len_of_oneTr)  # [1,10,182]，10是Batch size，182就是分类标签的onehot了

        output = output.squeeze(0)  # [10,182] 但是结果是未求softmax的。
        output = F.softmax(output, dim=1)  # dim表示在维度1上计算，因为维度0是batch。

        # 取预测最大值
        true_y = output.detach().cpu()  # 解绑梯度，才能被numpy之类操作。
        # 用pre_target保存一个batch的预测结果（top1的结果）。
        pre_target = np.argmax(true_y, axis=1)
        pre_target = torch.LongTensor(pre_target)
        # Y_vector保存一个batch的label。
        Y_vector = Y_vector.long()

        # 更新混淆矩阵CM
        real_one_batch_size = list(Y_vector.size())[0]
        for i in range(0, real_one_batch_size):
            row = "pre_" + str(pre_target[i].numpy())
            col = "label_" + str(Y_vector[i].numpy())
            CM.loc[row, col] = CM.loc[row, col] + 1

        # correct += (target == Y_vector).sum()

        # print("label:{}".format(Y_vector))
        # print("predict value:{}".format(target))

        # 更新topk准确率
        # np.argsort是从小到大排序，返回的坐标列表。
        orderedlist = np.argsort(true_y, axis=1)
        y_labels = Y_vector.numpy()

        top1_correct += count_topk(orderedlist, y_labels, top1)

        total += Y_vector.size(0)

    # print(CM)
    # 输出混淆矩阵产生的准确率、查准率、查全率（召回率）、F1-score
    # 调用cal_PRF计算P，R，F1
    precision_list, recall_list, MacroP, MacroR, MacroF1 = cal_PRF(CM, num_classes)

    return MacroP, MacroR, MacroF1, top1_correct / total


def test_model_withConfusionMatrix(model, test_data, batch_size, num_classes,logger):
    model.eval()
    # model.eval()#测试时，根本就没有反向传播调整参数的代码，因此不用担心测试的时候仍然会训练
    # model.train和eval这一对函数仅仅用于使用了BN和Dropout的场景下，训练集有一个dropout rate=0.5，
    # 而测试时要求dropout rate=1，所以通过train和eval两个函数通知框架，模型是处于什么状态，而选什么参数值。
    # 至于测试时，梯度是否在计算和累加，答案是如果不在测试阶段显式加入no_grad去掉梯度。因为测试阶段不需要反向传播梯度，这个计算是没用的。
    # 但是因为测试的代码里你没有写反向传播的代码，所以这个梯度的计算不会对模型产生影响，只是费一些内存而已。
    # 所以测试阶段加不加no_grad去掉梯度都可以。

    #####################计算准确率、召回率、F1-score
    # 创建多分类混淆矩阵（只针对top1的预测结果）
    ###          标签值
    #      ###类别1，类别2，类别3
    # 预  类别1
    # 测  类别2
    # 值  类别3
    pre_class = ["pre_" + str(i) for i in range(0, num_classes)]
    label_class = ["label_" + str(i) for i in range(0, num_classes)]
    CM = pd.DataFrame(index=pre_class, columns=label_class)
    CM.fillna(0, inplace=True)

    #########################计算topk准确率############
    #########计算Topk准确率
    top1 = 1
    top3 = 3
    top5 = 5
    top10 = 10

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0

    total = 0
    ######################################

    # 采用DataLoader从数据集中获取数据（Batch方式）
    test_data_inBatch = TrData(test_data)
    # 一个Batch取5条轨迹，测试集不shuffle。包括x和y，但在trainloader取出时会由collate_fn拆分
    test_loader = DataLoader(test_data_inBatch, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, drop_last=True)

    # print("开始预测！")

    for step, (X_vector, Y_vector, len_of_oneTr) in enumerate(test_loader):
        # 预测
        X_vector.cuda()
        output = model(X_vector, len_of_oneTr)  # [1,10,182]，10是Batch size，182就是分类标签的onehot了
        output = output.squeeze(0)  # [10,182] 但是结果是未求softmax的。
        output = F.softmax(output, dim=1)  # dim表示在维度1上计算，因为维度0是batch。

        # 取预测最大值
        true_y = output.detach().cpu()  # 解绑梯度，才能被numpy之类操作。
        # 用pre_target保存一个batch的预测结果（top1的结果）。
        pre_target = np.argmax(true_y, axis=1)
        pre_target = torch.LongTensor(pre_target)
        # Y_vector保存一个batch的label。
        Y_vector = Y_vector.long()

        # 更新混淆矩阵CM
        real_one_batch_size = list(Y_vector.size())[0]
        for i in range(0, real_one_batch_size):
            row = "pre_" + str(pre_target[i].numpy())
            col = "label_" + str(Y_vector[i].numpy())
            CM.loc[row, col] = CM.loc[row, col] + 1

        # correct += (target == Y_vector).sum()

        # print("label:{}".format(Y_vector))
        # print("predict value:{}".format(target))

        # 更新topk准确率
        # np.argsort是从小到大排序，返回的坐标列表。
        orderedlist = np.argsort(true_y, axis=1)
        y_labels = Y_vector.numpy()

        top1_correct += count_topk(orderedlist, y_labels, top1)
        top3_correct += count_topk(orderedlist, y_labels, top3)
        top5_correct += count_topk(orderedlist, y_labels, top5)
        top10_correct += count_topk(orderedlist, y_labels, top10)

        total += Y_vector.size(0)

    # print(CM)
    # CM.to_csv("ConfusionMatrix.csv", mode='w', index=True, header=True, encoding='utf_8_sig')
    # 输出混淆矩阵产生的准确率、查准率、查全率（召回率）、F1-score
    # 准确率或精度
    # 即混淆矩阵的对角线的值加和，除以总样本数
    Accracy = np.sum(np.array([CM.iloc[i, i] for i in range(0, num_classes)])) / total
    logger.info("Accuracy: {}".format(Accracy))

    # 调用cal_PRF计算P，R，F1
    precision_list, recall_list, MacroP, MacroR, MacroF1 = cal_PRF(CM, num_classes)
    # print("每个类别的查准率：{}".format(precision_list))
    # print("每个类别的查全率：{}".format(recall_list))
    logger.info("Macro-P: {}".format(MacroP))
    logger.info("Macro-R: {}".format(MacroR))
    logger.info("Macro-F1: {}".format(MacroF1))

    # 输出topk准确率
    logger.info('Test top{} Accuracy of the model on the testdata: {}'.format(top1, top1_correct / total))
    logger.info('Test top{} Accuracy of the model on the testdata: {}'.format(top3, top3_correct / total))
    logger.info('Test top{} Accuracy of the model on the testdata: {}'.format(top5, top5_correct / total))
    logger.info('Test top{} Accuracy of the model on the testdata: {}'.format(top10, top10_correct / total))


def toTenser(trainData, testData):
    Tr_after_group = trainData.groupby(['userID', 'TrID'], as_index=False)
    Tr_groups = Tr_after_group.count()
    len_of_Tr_g = len(Tr_groups)
    train_data = []
    # 循环取每条轨迹，每条轨迹构建一个Tensor，然后加入train_data这个list中
    for tr in range(0, len_of_Tr_g):
        oneTr = Tr_after_group.get_group((Tr_groups.iloc[tr]['userID'], Tr_groups.iloc[tr]['TrID']))
        # print(oneTr)
        oneTr.drop(columns=['TrID', 'rowID', 'colID', 'utc', 'lid'], inplace=True)
        train_data.append(torch.Tensor(oneTr.values))

    Te_after_group = testData.groupby(['userID', 'TrID'], as_index=False)
    Te_groups = Te_after_group.count()
    len_of_Te_g = len(Te_groups)
    test_data = []
    for te in range(0, len_of_Te_g):
        oneTr = Te_after_group.get_group((Te_groups.iloc[te]['userID'], Te_groups.iloc[te]['TrID']))
        # print(oneTr)
        oneTr.drop(columns=['TrID', 'rowID', 'colID', 'utc', 'lid'], inplace=True)
        test_data.append(torch.Tensor(oneTr.values))

    return train_data, test_data


# 对每个batch的结果，从排序后的预测结果中，获得topk的统计，作为累加的单次值
def count_topk(orderedlist, y_labels, topk):
    topk_result = orderedlist[:, -topk:]
    rl = []
    tk = topk_result.numpy()

    for t in range(0, len(tk)):
        if y_labels[t] in tk[t]:
            rl.append(1)
        else:
            rl.append(0)
    r_count = np.sum(np.array(rl))
    return r_count


# 对混淆矩阵进行P，R，F1的计算
def cal_PRF(CM, num_classes):
    # 多分类的查准率和查全率用的是Macro，即宏查准率、宏查全率和宏F1
    # 即计算每个类别的查准率、查全率求一个均值得到宏查准率、宏查全率
    # 再用宏查准率、宏查全率计算宏F1

    # 查准率Precision计算
    precision_list = []
    # 查全率Recall
    recall_list = []
    for i in range(0, num_classes):
        TP = CM.iloc[i, i]  # 类i的真正例
        FP = CM.iloc[i, :].sum() - TP  # 假正例，也就是用预测为正例的个数减去真正例个数
        if TP == 0 and FP == 0:
            precision = 0
        elif FP == 0:
            precision = 1
        else:
            precision = TP / (TP + FP)
        precision_list.append(precision)
        recall = TP / CM.iloc[:, i].sum()  # 召回率是用真正例个数除以样本中label为正例的个数。
        recall_list.append(recall)
    recall_list = np.nan_to_num(recall_list)
    MacroP = np.sum(np.array(precision_list)) / num_classes
    MacroR = np.sum(np.array(recall_list)) / num_classes
    MacroF1 = 2 * MacroP * MacroR / (MacroP + MacroR)

    return precision_list, recall_list, MacroP, MacroR, MacroF1


def statistic(Trs):
    userNum = len(Trs['userID'].unique())
    tmpG = Trs.groupby(['userID', 'TrID'], as_index=False)
    count_table = tmpG.size().rename(columns={'size': 'lenofTr'})
    TrNum = len(count_table)
    return TrNum, userNum


def main():
    print('TUL checker...')
    args = parse_args()
    logger = getLogger(args.dataset, args.isDense)
    logger.info('configure:')
    logger.info('   batchSize: {}'.format(args.batchSize))
    logger.info('   epochs: {}'.format(args.epochs))
    logger.info('   numHeads: {}'.format(args.numHeads))

    # load checkin data
    if args.isDense == 'D':
        dataPath = './data/' + args.dataset + '/dense/AllsubTr.csv'
    elif args.isDense == 'S':
        dataPath = './data/' + args.dataset + '/sparse/AllsubTr.csv'
    else:
        print('error argument: isDense')
        exit(0)
    trajectories = pd.read_csv(dataPath)
    TrNum, userNum = statistic(trajectories)
    logger.info('num of trajectory: {}\nnum of user: {}'.format(TrNum, userNum))

    # split train-test dataset
    trainSet, testSet, numTrain, numTest = splitData(trajectories, args.testNum)
    logger.info('num of trajectory in trainSet: {}'.format(numTrain))
    logger.info('num of trajectory in testSet: {}'.format(numTest))

    # embed
    trainData, testData, onehotSize = approximateOnehotEmbed(trajectories, trainSet, testSet)
    logger.info('dimension of approximateOnehot: {}'.format(onehotSize))

    # build model
    trainData, testData = toTenser(trainData, testData)
    oneTr = trainData[0]
    onePointSize = oneTr.size(1)
    inputSize = onePointSize - 1
    num_classes = max(trajectories['userID'].unique()) + 1
    model = Multi_Head_Attention_LSTM(input_size=inputSize,
                                      hidden_size=args.hiddenSize,
                                      num_layers=1,
                                      num_classes=num_classes,
                                      batch_size=args.batchSize,
                                      num_heads=args.numHeads)
    model.cuda()

    # train
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    mymodel, losslist, acclist = train_model(model, trainData, args.batchSize,
                                             args.epochs, args.lr, testData, num_classes)
    end_time.record()
    torch.cuda.synchronize()
    logger.info("train done.\ntime cost: {}".format(start_time.elapsed_time(end_time)))

    # test
    test_model_withConfusionMatrix(mymodel, testData, batch_size=args.batchSize, num_classes=num_classes,logger=logger)

    logger.info('Done')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
