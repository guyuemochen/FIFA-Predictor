import keras
import itertools
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
        for i in t1:
            for j in t2:
                dataset.append([i[x] + j[x] for x in range(len(i))])
                dataset.append([j[x] + i[x] for x in range(len(i))])
                if data[2] > data[3]:
                    label.append([1, 0, 0])
                    label.append([0, 0, 1])
                else:
                    label.append([0, 1])
                    label.append([1, 0])
    return np.array(dataset), np.array(label)


def getData(t1, t2):
    team1, team2 = getTwoCountries(t1, t2)
    data = [team1[0][i] + team2[0][i] for i in range(7)]
    return np.array([data])


if __name__ == '__main__':
    reconstructed_model = keras.models.load_model("FIFAPredict")
    t1, t2 = input('请输入球队编号（见Datas中countries.txt文件）：').split()
    t1, t2 = int(t1), int(t2)
    x = getData(t1, t2)
    result = reconstructed_model.predict(x, verbose=0)
    result = np.array(result[0])
    result = result / sum(result)
    print(f'队伍1获胜概率为{round(result[0] * 100, 2)}\n队伍2获胜概率为{round(result[2] * 100, 2)}\n进入加时赛概率为{round(result[1] * 100, 2)}')