import numpy as np
import itertools
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.client import device_lib
from keras import Sequential
from keras.layers import LSTM, Dense, Input


def readFile(path):
    '''
    读取之前比分文件
    :param path: 读取文件路径
    :return:
    '''
    file = open(path, "r")
    lines = file.read().split('\n')
    dataList = []
    for line in lines:
        datas = line.split(' ')
        for i, data in enumerate(datas):
            datas[i] = int(data)
        dataList.append(datas)
    return dataList


def getTwoCountries(team1, team2, dataList=None):
    if dataList is None:
        dataList = readFile('./Datas/soccer.txt')
    team1_set = []
    team2_set = []
    for data in dataList:
        if data[0] == team1 and data[-1] == 1:
            team1_set.append([data[2], data[3]])
        if data[1] == team1 and data[-1] == 1:
            team1_set.append([data[3], data[2]])
        if data[0] == team2 and data[-1] == 1:
            team2_set.append([data[2], data[3]])
        if data[1] == team2 and data[-1] == 1:
            team2_set.append([data[3], data[2]])
    team1_set = list(itertools.permutations(team1_set))
    team2_set = list(itertools.permutations(team2_set))
    team1_later = {2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0]}
    team2_later = {2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0]}
    for data in dataList:
        if data[-1] > 1:
            if data[0] == team1:
                team1_later[data[-1]][0] = data[2]
                team1_later[data[-1]][1] = data[3]
            if data[1] == team1:
                team1_later[data[-1]][0] = data[3]
                team1_later[data[-1]][1] = data[2]
            if data[0] == team2:
                team2_later[data[-1]][0] = data[2]
                team2_later[data[-1]][1] = data[3]
            if data[1] == team2:
                team2_later[data[-1]][0] = data[3]
                team2_later[data[-1]][1] = data[2]
    team1_set = [list(x) + list(team1_later.values()) for x in team1_set]
    team2_set = [list(x) + list(team2_later.values()) for x in team2_set]
    return team1_set, team2_set


def getDatasetAndLabels():
    dataList = readFile('./Datas/soccer.txt')
    dataset = []
    label = []
    for data in dataList:
        t1, t2 = getTwoCountries(data[0], data[1], dataList)
        if data[2] == data[3]:
            continue
        for i in t1:
            for j in t2:
                if data[2] > data[3]:
                    dataset.append([i[x] + j[x] for x in range(len(i))])
                    dataset.append([j[x] + i[x] for x in range(len(i))])
                    label.append([1, 0])
                    label.append([0, 1])
                elif data[2] > data[3]:
                    dataset.append([i[x] + j[x] for x in range(len(i))])
                    dataset.append([j[x] + i[x] for x in range(len(i))])
                    label.append([0, 1])
                    label.append([1, 0])
    return np.array(dataset), np.array(label)


if __name__ == '__main__':
    try:
        model = keras.models.load_model("FIFAPredict")
    except Exception:
        print('here')
        model = Sequential()
        model.add(Input(shape=(7, 4)))
        model.add(LSTM(4, dropout=0.2))
        model.add(Dense(2, activation='sigmoid'))

    model.summary()

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy']
    )

    x, y = getDatasetAndLabels()
    print(x.shape)
    model.fit(x, y, epochs=100, batch_size=100)
    model.save('FIFAPredict')