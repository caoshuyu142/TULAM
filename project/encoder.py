import warnings

import pandas as pd
from sklearn import preprocessing

warnings.filterwarnings('ignore')


def addLIDtoUserSubT(allTr):
    # 输入所有子轨迹
    allSubT = allTr
    allSubT['rowID'] = allSubT['rowID'].astype(int)
    allSubT['colID'] = allSubT['colID'].astype(int)
    allSubT['rowID'] = allSubT['rowID'].astype(str)
    allSubT['colID'] = allSubT['colID'].astype(str)

    allSubT['lid'] = allSubT[['rowID', 'colID']].apply(lambda x: 'X'.join(x), axis=1)
    return allSubT


# lid直接onehot会让onehot维度太多。
# 两个列分别one-hot，然后拼接起来，即两个属性列，合并起来的标识中，会有两个位为1
# 这种方式会让onehot维度降低
# 并且将onehot作为多列存储。
def onehotEncoding3(allSubTr):
    tempdata = allSubTr[['rowID', 'colID']]
    #print(tempdata.head())
    #print("******************************")
    # tempdata1 = allSubTr['lid']
    # print(tempdata)
    # print(tempdata1)

    # 调用sklearn完成编码
    enc = preprocessing.OneHotEncoder()
    enc.fit(tempdata)  # enc已经训练好了

    # transform是可以一次转化多个标签的。
    # tmp = enc.transform([['233','183']]).toarray()
    # print(tmp[0])
    # print(tmp[1])
    # print(tmp)
    # onehot_size = len(tmp[0])
    # print("onehot向量的长度为:{}".format(onehot_size))

    # 构建一个新的DF，一列是lid，后面都是onehot0,onehot1......onehotn
    onehot = enc.transform(tempdata).toarray()

    onehot_size = len(onehot[0])

    # 构建一个新的DF，一列是lid，后面都是onehot0,onehot1......onehotn
    colnewname = ['onehot_' + str(i) for i in range(0, onehot_size)]
    newDF = pd.DataFrame(onehot, columns=colnewname)
    # onehot.rename(columns=dict(colnewname),inplace = True)
    # print(newDF)
    newDF['lid'] = allSubTr['lid']
    # print(newDF)
    # print(len(newDF))
    # print(newDF.dtypes)
    # newDF有7816行，说明有冗余重复的。因为网格的行是300多，列是500多，应该共800多个onehot
    newDF = newDF.drop_duplicates('lid')
    # print(len(newDF)) #1305

    return newDF, onehot_size


def onehotonData(allTr, tr, onehot_size):
    colnewname = ['onehot_' + str(i) for i in range(0, onehot_size)]
    dataheader = ['userID', 'TrID', "rowID", "colID", "utc", 'lid']
    newdataheader = dataheader + colnewname

    tr = addLIDtoUserSubT(tr)
    tr['userID'] = tr['userID'].astype('int16')
    tr['TrID'] = tr['TrID'].astype('int32')

    tr = pd.merge(tr, allTr, how='left', on=['lid'])
    return tr


def approximateOnehotEmbed(allTr,trainSet,testSet):
    allTr = addLIDtoUserSubT(allTr)
    allTr, onehotSize = onehotEncoding3(allTr)
    trainSetinOnehot = onehotonData(allTr,trainSet,onehotSize)
    testSetinOnehot = onehotonData(allTr,testSet,onehotSize)
    return trainSetinOnehot, testSetinOnehot, onehotSize
