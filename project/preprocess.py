import random
import time
import pandas as pd
import numpy as np
import math
from tqdm import tqdm


def rename_time_foursquare(DF):
    dt_time = time.strptime(DF['utc'], '%a %b %d %H:%M:%S +0000 %Y')
    timestamp = int(time.mktime(dt_time))
    return timestamp


def rename_time_gowalla(DF):
    dt_time = time.strptime(DF['utc'], '%Y-%m-%dT%H:%M:%SZ')
    timestamp = int(time.mktime(dt_time))
    return timestamp


def rename_uid(DF):
    userID_list = DF['userID'].unique()
    newid = 0
    DF['newuserID'] = np.NaN
    for uid in userID_list:
        tmp_index = DF[DF['userID'] == uid].index.tolist()
        DF.loc[tmp_index, 'newuserID'] = newid
        newid = newid + 1

    DF['userID'] = DF['newuserID']
    DF.drop(columns=['newuserID'], inplace=True)
    DF['userID'] = DF['userID'].astype(int)
    return DF


def narrowScope(Allcheckins, minlon=-180.0, maxlon=180.0, minlat=-85.0, maxlat=85.0, threshold_checkins=100):
    Removelist = Allcheckins[Allcheckins['longitude'] < minlon].index.tolist()
    Removelist += Allcheckins[Allcheckins['longitude'] > maxlon].index.tolist()
    Removelist += Allcheckins[Allcheckins['latitude'] < minlat].index.tolist()
    Removelist += Allcheckins[Allcheckins['latitude'] > maxlat].index.tolist()
    Allcheckins.drop(Removelist, axis=0, inplace=True)
    g = Allcheckins.groupby(['userID'], as_index=False)

    count_table = g.size().rename(columns={'size': 'num_checkins'})
    retain_user = count_table[count_table['num_checkins'] >= threshold_checkins]
    retain_list = retain_user['userID'].tolist()

    Allcheckins = Allcheckins[Allcheckins['userID'].isin(retain_list)]
    return Allcheckins


def getData(dataset_path, minlon, maxlon, minlat, maxlat, threshold_checkins):
    print('----------load dataset...----------')

    # ---------------read dataset--------------#
    allCheckin = pd.read_csv(dataset_path)

    # -------------narrow the scope----------#
    allCheckin = narrowScope(allCheckin, minlon, maxlon, minlat, maxlat, threshold_checkins)

    # -------------statistic--------------#
    checkinNum = len(allCheckin)
    userNum = len(allCheckin['userID'].drop_duplicates().values.tolist())

    allCheckin = rename_uid(allCheckin)
    allCheckin.reset_index(inplace=True, drop=True)

    return allCheckin, checkinNum, userNum


def txt2csv_foursquare(filename):
    header = ['userID', 'latitude', 'longitude', 'utc']
    rst = pd.DataFrame(columns=header)
    reader = pd.read_table(filename, header=None, skiprows=0, sep='\t', iterator=True, encoding='ISO-8859-1',
                           error_bad_lines=False)
    loop = True
    i = 0
    batchSize = 1000000
    while loop:
        i = i + 1
        try:
            chunk = reader.get_chunk(batchSize)
            chunk_filtered = pd.DataFrame(chunk, columns=[0, 4, 5, 7])
            chunk_filtered.columns = header
            rst = pd.concat([rst, chunk_filtered])
        except StopIteration:
            loop = False
            pass
        pass

    rst['utc'] = rst.apply(rename_time_foursquare, axis=1)
    rename_uid(rst)
    return rst


def txt2csv_gowalla(filename):
    header = ['userID', 'latitude', 'longitude', 'utc']
    rst = pd.DataFrame(columns=header)
    reader = pd.read_table(filename, header=None, skiprows=0, sep='\t', iterator=True, encoding='ISO-8859-1',
                           error_bad_lines=False)
    loop = True
    i = 0
    batchSize = 1000000
    while loop:
        i = i + 1
        try:
            chunk = reader.get_chunk(batchSize)
            chunk_filtered = pd.DataFrame(chunk, columns=[0, 2, 3, 1])
            chunk_filtered.columns = header
            rst = pd.concat([rst, chunk_filtered])
        except StopIteration:
            loop = False
            pass
        pass
    rst['utc'] = rst.apply(rename_time_gowalla, axis=1)
    rename_uid(rst)
    return rst


def coordinate2grid(Allcheckins, cellsize: float) -> pd.DataFrame:
    maxlon, minlon = Allcheckins["longitude"].max(), Allcheckins["longitude"].min()
    maxlat, minlat = Allcheckins["latitude"].max(), Allcheckins["latitude"].min()
    midlat = (maxlat + minlat) / 2
    km_per_lon, km_per_lat = 111.32 * math.cos(math.radians(midlat)), 111
    deg_lon, deg_lat = cellsize / km_per_lon, cellsize / km_per_lat
    lon_grids, lat_grids = int((maxlon - minlon) / deg_lon) + 1, int((maxlat - minlat) // deg_lat) + 1
    grid_lon, grid_lat = (maxlon - minlon) / lon_grids, (maxlat - minlat) / lat_grids
    Allcheckins["longitude"] = Allcheckins["longitude"].sub(minlon).floordiv(grid_lon).astype(int)
    Allcheckins["latitude"] = Allcheckins["latitude"].sub(minlat).floordiv(grid_lat).astype(int)
    Allcheckins = Allcheckins.rename(columns={"longitude": "colID", "latitude": "rowID"})
    return Allcheckins


def checkin2trajectory(Allcheckins, interval_hour):
    RawSubTr = pd.DataFrame(columns=['userID', 'TrID', 'rowID', 'colID', 'utc'])
    userNum = len(Allcheckins['userID'].unique().tolist())
    TrNum = 0
    for userID in tqdm(range(userNum), desc='divide checkins into trajectory'):
        Tr_user = Allcheckins[Allcheckins['userID'] == userID]
        Tr_user = Tr_user.sort_values(by="utc", ascending=True)
        Tr_user_sorted = Tr_user.reset_index(drop=True)
        TrID = 0

        for index in range(0, len(Tr_user_sorted)):
            if index != len(Tr_user_sorted) - 1:
                intervalSec = Tr_user_sorted.iloc[index + 1]['utc'] - Tr_user_sorted.iloc[index]['utc']
                rowID = Tr_user_sorted.iloc[index]['rowID']
                colID = Tr_user_sorted.iloc[index]['colID']
                ts = Tr_user_sorted.iloc[index]['utc']
                one_point = np.array([userID, TrID, rowID, colID, ts]).reshape(1, 5)
                insertRow = pd.DataFrame(one_point, columns=['userID', 'TrID', 'rowID', 'colID', 'utc'])
                RawSubTr = RawSubTr.append(insertRow, ignore_index=True)
                if intervalSec >= interval_hour * 60 * 60:
                    TrID = TrID + 1
            else:
                rowID = Tr_user_sorted.iloc[index]['rowID']
                colID = Tr_user_sorted.iloc[index]['colID']
                ts = Tr_user_sorted.iloc[index]['utc']
                one_point = np.array([userID, TrID, rowID, colID, ts]).reshape(1, 5)
                insertRow = pd.DataFrame(one_point, columns=['userID', 'TrID', 'rowID', 'colID', 'utc'])
                RawSubTr = RawSubTr.append(insertRow, ignore_index=True)
                TrNum = TrNum + TrID + 1
                pass
            pass
        pass
    return RawSubTr, TrNum


def splitData(allTrajectoryData, testNum):
    trainSet = pd.DataFrame(columns=['userID', 'TrID', 'rowID', 'colID', 'utc'])
    testSet = pd.DataFrame(columns=['userID', 'TrID', 'rowID', 'colID', 'utc'])

    userList = allTrajectoryData['userID'].unique()
    for u in userList:
        userTrs = allTrajectoryData[allTrajectoryData['userID']==u]
        trList = userTrs['TrID'].unique()
        random.shuffle(trList)
        totalNum = len(trList)
        trainNum = totalNum - testNum
        trainList = trList[:trainNum]
        testList = trList[trainNum:]
        trainData = userTrs[userTrs['TrID'].isin(trainList)]
        testData = userTrs[userTrs['TrID'].isin(testList)]
        trainSet = trainSet.append(trainData, ignore_index=True)
        testSet = testSet.append(testData, ignore_index=True)

    tmpG = trainSet.groupby(['userID','TrID'],as_index=False)
    count_table = tmpG.size().rename(columns={'size': 'lenofTr'})
    numTrain = len(count_table)
    tmpG = testSet.groupby(['userID','TrID'],as_index=False)
    count_table = tmpG.size().rename(columns={'size': 'lenofTr'})
    numTest = len(count_table)
    return trainSet, testSet, numTrain, numTest


