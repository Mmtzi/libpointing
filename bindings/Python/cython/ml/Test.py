import numpy as np


myData = [0, 1, 0, 1, 0, 1, -300, 100, -300 , 100, -300, 100]
mySecondData = [[0,1],[1,0], [4,0]]

mynewList = []
mynewList.append(myData[0])
mynewList.append(mySecondData[0])
print(mynewList)
mynewList = []
mynewList.append([myData[0], mySecondData[0]])
print(mynewList)
bla = np.zeros((20, 4, 1))
blo = np.empty((20, 4, 1))
bla = blo
print(bla)