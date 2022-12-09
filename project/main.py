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
    onetr = trs[0]
    onepoint_size = onetr.size(1)

    input_size = onepoint_size - 1

    trs.sort(key=lambda x: len(x), reverse=True)

    tr_lengths = [len(sq) for sq in trs]

    trs = rnn_utils.pad_sequence(trs, batch_first=True, padding_value=0)

    var_x = trs[:, :, 1:input_size + 1]  # 在第3个维度上取1-XXX
    tmpy = trs[:, :, 0]

    var_y = tmpy[:, 0]

    return var_x, var_y, tr_lengths


def train_model(model, train_data, batch_size, num_epochs, learning_rate, test_data, num_classes):
    train_data_inBatch = TrData(train_data)

    train_loader = DataLoader(train_data_inBatch, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

    losslist = []
    acclist = []
    for epoch in tqdm(range(num_epochs),desc='Training...'):
        model.train()
        ave_loss_of_epoch = 0
        count = 0
        for step, (X_vector, Y_vector, len_of_oneTr) in enumerate(train_loader):

            X_vector.cuda()
            Y_vector.cuda()

            out = model(X_vector, len_of_oneTr)

            out = out.squeeze(0).cuda()

            Y_vector = Y_vector.long().cuda()
            loss = criterion(out, Y_vector)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ave_loss_of_epoch = ave_loss_of_epoch + loss
            count = count + 1

        if optimizer.param_groups[0]['lr'] > 0.0001:
            scheduler.step()

        ave_loss_of_epoch = ave_loss_of_epoch / count
        losslist.append(ave_loss_of_epoch.item())
        MacroP, MacroR, MacroF1, top1_correct_Rate = test_model_inTrain(model, test_data, batch_size, num_classes)

        acclist.append(top1_correct_Rate)

    return model, losslist, acclist


def test_model_inTrain(model, test_data, batch_size, num_classes):
    model.eval()
    pre_class = ["pre_" + str(i) for i in range(0, num_classes)]
    label_class = ["label_" + str(i) for i in range(0, num_classes)]
    CM = pd.DataFrame(index=pre_class, columns=label_class)
    CM.fillna(0, inplace=True)

    top1 = 1

    top1_correct = 0

    total = 0

    test_data_inBatch = TrData(test_data)

    test_loader = DataLoader(test_data_inBatch, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, drop_last=True)


    for step, (X_vector, Y_vector, len_of_oneTr) in enumerate(test_loader):
        X_vector.cuda()
        output = model(X_vector, len_of_oneTr) 

        output = output.squeeze(0) 
        output = F.softmax(output, dim=1) 

        true_y = output.detach().cpu() 
        pre_target = np.argmax(true_y, axis=1)
        pre_target = torch.LongTensor(pre_target)

        Y_vector = Y_vector.long()

        real_one_batch_size = list(Y_vector.size())[0]
        for i in range(0, real_one_batch_size):
            row = "pre_" + str(pre_target[i].numpy())
            col = "label_" + str(Y_vector[i].numpy())
            CM.loc[row, col] = CM.loc[row, col] + 1

        orderedlist = np.argsort(true_y, axis=1)
        y_labels = Y_vector.numpy()

        top1_correct += count_topk(orderedlist, y_labels, top1)

        total += Y_vector.size(0)

    precision_list, recall_list, MacroP, MacroR, MacroF1 = cal_PRF(CM, num_classes)

    return MacroP, MacroR, MacroF1, top1_correct / total


def test_model_withConfusionMatrix(model, test_data, batch_size, num_classes,logger):
    model.eval()
    pre_class = ["pre_" + str(i) for i in range(0, num_classes)]
    label_class = ["label_" + str(i) for i in range(0, num_classes)]
    CM = pd.DataFrame(index=pre_class, columns=label_class)
    CM.fillna(0, inplace=True)

    top1 = 1
    top3 = 3
    top5 = 5
    top10 = 10

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0

    total = 0

    test_data_inBatch = TrData(test_data)

    test_loader = DataLoader(test_data_inBatch, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, drop_last=True)

    for step, (X_vector, Y_vector, len_of_oneTr) in enumerate(test_loader):
        X_vector.cuda()
        output = model(X_vector, len_of_oneTr)
        output = output.squeeze(0)
        output = F.softmax(output, dim=1)

        true_y = output.detach().cpu() 

        pre_target = np.argmax(true_y, axis=1)
        pre_target = torch.LongTensor(pre_target)

        Y_vector = Y_vector.long()

        real_one_batch_size = list(Y_vector.size())[0]
        for i in range(0, real_one_batch_size):
            row = "pre_" + str(pre_target[i].numpy())
            col = "label_" + str(Y_vector[i].numpy())
            CM.loc[row, col] = CM.loc[row, col] + 1

        orderedlist = np.argsort(true_y, axis=1)
        y_labels = Y_vector.numpy()

        top1_correct += count_topk(orderedlist, y_labels, top1)
        top3_correct += count_topk(orderedlist, y_labels, top3)
        top5_correct += count_topk(orderedlist, y_labels, top5)
        top10_correct += count_topk(orderedlist, y_labels, top10)

        total += Y_vector.size(0)

    Accracy = np.sum(np.array([CM.iloc[i, i] for i in range(0, num_classes)])) / total
    logger.info("Accuracy: {}".format(Accracy))

    precision_list, recall_list, MacroP, MacroR, MacroF1 = cal_PRF(CM, num_classes)

    logger.info("Macro-P: {}".format(MacroP))
    logger.info("Macro-R: {}".format(MacroR))
    logger.info("Macro-F1: {}".format(MacroF1))

    logger.info('Test top{} Accuracy of the model on the testdata: {}'.format(top1, top1_correct / total))
    logger.info('Test top{} Accuracy of the model on the testdata: {}'.format(top3, top3_correct / total))
    logger.info('Test top{} Accuracy of the model on the testdata: {}'.format(top5, top5_correct / total))
    logger.info('Test top{} Accuracy of the model on the testdata: {}'.format(top10, top10_correct / total))


def toTenser(trainData, testData):
    Tr_after_group = trainData.groupby(['userID', 'TrID'], as_index=False)
    Tr_groups = Tr_after_group.count()
    len_of_Tr_g = len(Tr_groups)
    train_data = []

    for tr in range(0, len_of_Tr_g):
        oneTr = Tr_after_group.get_group((Tr_groups.iloc[tr]['userID'], Tr_groups.iloc[tr]['TrID']))
        oneTr.drop(columns=['TrID', 'rowID', 'colID', 'utc', 'lid'], inplace=True)
        train_data.append(torch.Tensor(oneTr.values))

    Te_after_group = testData.groupby(['userID', 'TrID'], as_index=False)
    Te_groups = Te_after_group.count()
    len_of_Te_g = len(Te_groups)
    test_data = []
    for te in range(0, len_of_Te_g):
        oneTr = Te_after_group.get_group((Te_groups.iloc[te]['userID'], Te_groups.iloc[te]['TrID']))
        oneTr.drop(columns=['TrID', 'rowID', 'colID', 'utc', 'lid'], inplace=True)
        test_data.append(torch.Tensor(oneTr.values))

    return train_data, test_data


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


def cal_PRF(CM, num_classes):
    precision_list = []
    recall_list = []
    for i in range(0, num_classes):
        TP = CM.iloc[i, i] 
        FP = CM.iloc[i, :].sum() - TP 
        if TP == 0 and FP == 0:
            precision = 0
        elif FP == 0:
            precision = 1
        else:
            precision = TP / (TP + FP)
        precision_list.append(precision)
        recall = TP / CM.iloc[:, i].sum()
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


if __name__ == '__main__':
    main()
