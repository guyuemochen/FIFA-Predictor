import torch.nn as nn
import torch
import itertools

class SoccerGuess(nn.Module):

    def __init__(self) -> None:
        super(SoccerGuess, self).__init__()
        self.layer1 = nn.Linear(24, 3)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        output = self.layer1(x)
        output = self.sig(output)
        return output

def getData(team1, team2, until = 1):
    t1, t2 = getTwoCountriesUntil(team1, team2, until)
    return torch.FloatTensor(t1[0]+t2[0])

def getTwoCountriesUntil(team1, team2, index = None):
    team1_set, team2_set = getTwoCountries(team1, team2)
    if index is None:
        return team1_set, team2_set
    if index < 1 or index > 4:
        return None
    for i in range(4 + index * 2, 12):
        for j in range(len(team1_set)):
            team1_set[j][i] = 0
        for j in range(len(team2_set)):
            team2_set[j][i] = 0
    return team1_set, team2_set

def readFile(path):
    file = open(path, "r")
    lines = file.read().split('\n')
    dataList = []
    for line in lines:
        datas = line.split(' ')
        for i, data in enumerate(datas):
            datas[i] = int(data)
        dataList.append(datas)
    return dataList

def getTwoCountries(team1, team2):
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
    team1_set = [i[0]+i[1]+i[2] for i in team1_set]
    team2_set = [i[0]+i[1]+i[2] for i in team2_set]
    team1_after = {}
    team2_after = {}
    for data in dataList:
        if data[0] == team1 and data[-1] == 2:
            team1_after[2] = [data[2], data[3]]
        if data[1] == team1 and data[-1] == 2:
            team1_after[2] = [data[3], data[2]]
        if data[0] == team2 and data[-1] == 2:
            team2_after[2] = [data[2], data[3]]
        if data[1] == team2 and data[-1] == 2:
            team2_after[2] = [data[3], data[2]]
    for i in range(2, 5):
        if i in team1_after:
            team1_set = [set + team1_after[i] for set in team1_set]
        else:
            team1_set = [set + [0, 0] for set in team1_set]
        if i in team2_after:
            team2_set = [set + team2_after[i] for set in team2_set]
        else:
            team2_set = [set + [0, 0] for set in team2_set]
    return team1_set, team2_set


if __name__ == '__main__':
    model = SoccerGuess()
    model.load_state_dict(torch.load('soccermodel.md'))
    model.eval()
    t1, t2 = input('请输入球队编号（见Datas中countries.txt文件）：').split()
    t1, t2 = int(t1), int(t2)
    result = list(model(getData(t1, t2)).data)
    result = [i / sum(result) for i in result]
    print(f'team1: {round(float(result[0] * 100), 2)}%\nteam2: {round(float(result[2] * 100), 2)}%\ntie  : {round(float(result[1] * 100), 2)}%')