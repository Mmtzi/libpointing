import numpy as np

#import calcScore
myData = []
for i in range (0,20):
    myData.append((0, 1, 0, 1, 0, 1, -300, 100, -300 , 100, -300, 100))
#print(myData)
myFirstData = np.array([[[0,0], [1,0], [3,0]],[[0,0], [1,0], [3,0]]])
mySecondData = np.array([[[0,1],[1,0], [4,0]],[[0,1],[1,0], [4,0]]])
print(myFirstData)
print(myFirstData[1][0][0])
#if myFirstData[0][1] == mySecondData[0][1]:
    #print("yeah")

mynewList = []
mynewList.append(myData[0])
mynewList.append(mySecondData[0])
#print(mynewList)
mynewList = []
mynewList.append([myData[0], mySecondData[0]])
#print(mynewList)
bla = np.zeros((20, 4, 1))
blo = np.empty((20, 4, 1))
bla = blo
#print(bla)

actCursorPos= 1000,1000
oldCursorPos = 1000, 993
olddistance = 507
distance = 500
targetPosition = 1000, 1500
targetSize = 30
#calcScore.calcScoreOfAction(actCursorPos, oldCursorPos, distance, olddistance, targetPosition, targetSize)

lr=0.001


myarray = np.random.randint(16, size=(4,4))
bla = [[[5,4],[3,2]],[[1,2],[7,8]]]
bla = np.array(bla)
print(bla)

mean = np.mean(bla, axis=1)
st = np.std(bla,axis=1)
print(mean)
print(st)
bla = np.subtract(bla, mean)
print(bla)
bla = np.divide(bla, st)
print(bla)

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]


def lrrek(lr):
    lr = lr*(1-lr*100)
    print(lr)
    if lr >0.00001:
        lrrek(lr)
#lrrek(lr)