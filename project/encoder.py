import warnings

import pandas as pd
from sklearn import preprocessing

warnings.filterwarnings('ignore')


def addLIDtoUserSubT(allTr):
    allSubT = allTr
    allSubT['rowID'] = allSubT['rowID'].astype(int)
    allSubT['colID'] = allSubT['colID'].astype(int)
    allSubT['rowID'] = allSubT['rowID'].astype(str)
    allSubT['colID'] = allSubT['colID'].astype(str)

    allSubT['lid'] = allSubT[['rowID', 'colID']].apply(lambda x: 'X'.join(x), axis=1)
    return allSubT


def onehotEncoding3(allSubTr):
    tempdata = allSubTr[['rowID', 'colID']]

    enc = preprocessing.OneHotEncoder()
    enc.fit(tempdata)

    onehot = enc.transform(tempdata).toarray()

    onehot_size = len(onehot[0])

    # 构建一个新的DF，一列是lid，后面都是onehot0,onehot1......onehotn
    colnewname = ['onehot_' + str(i) for i in range(0, onehot_size)]
    newDF = pd.DataFrame(onehot, columns=colnewname)

    newDF['lid'] = allSubTr['lid']
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
