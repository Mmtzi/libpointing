from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers
from keras.layers.recurrent import LSTM, GRU
from numpy import genfromtxt
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import os
import numpy as np
from threading import Thread
from queue import Queue
import time

class Simulator(Thread):
    def __init__(self, q):
        self.epochs = 10
        self.dataQueue = q
        self.sleepInterval = 0.2
        self.pastTimeSteps = 20
        self.trainSize = 500
        self.validSize = 100
        self.pastDxDyList = [0]*self.pastTimeSteps*2
        super().__init__()


    # def create_dataset(self, dataset):
    #     dataX = []
    #     for i in range(len(dataset) - self.look_back + 1):
    #         a = dataset[i:(i + self.look_back), :]
    #         dataX.append(a)
    #         #print(a)
    #     return np.array(dataX)

    def run(self):
        while True:
            # load Data
            # 'dx', 'dy', 'rx', 'ry', 'button', 'time', 'distance', 'directionX', 'directionY',
            # 'targetX', 'targetY', 'targetSize', 'initMouseX', 'initMouseY', 'targetID'

            myTrainDataList = []
            myValidDataList = []
            sleepTime = 0
            print(self.dataQueue.qsize())

            while self.dataQueue.qsize() <= (self.trainSize +self.validSize):
                time.sleep(self.sleepInterval)
                sleepTime +=self.sleepInterval


            self.createTrainSet(myTrainDataList)

            myTrainDataList = np.array(myTrainDataList)
            myTrainInData = myTrainDataList[:, 9:56]
            myTrainOutData = myTrainDataList[:,0:2]

            # define and fit the final model

            if os.path.exists('ml\\models/sim_adv_lr_model.h5'):
                model = load_model('ml\\models/sim_adv_lr_model.h5')
                print("loaded model")
            else:
                model = Sequential()
                model.add(Dense(16, input_dim=46, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
                model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
                model.add(Dense(2, activation='linear'))
                model.compile(Adam(lr=0.001), loss='mse')

            model.fit(myTrainInData, myTrainOutData, epochs=self.epochs, verbose=2, batch_size=20, shuffle=True)
            model.save('ml\\models\\sim_adv_lr_model.h5')

            self.predictMyData(model)

            predictedDX, realDX = self.predictTrainData(model, myValidDataList)

            #allDX = []
            #predictedAllDX = []
            #allDX = np.arange(-127, 127, 1)
            #predictedAllDX = model.predict(allDX)
            # plotDX = []
            # plotPred = []
            #
            # for j in range(len(allDX)):
            #     plotDX.append(predictedAllDX[j][0])
            #     plotPred.append(allDX[j][0])

            self.plotResults(predictedDX, realDX)

    def plotResults(self, predictedDX, realDX):
        print("ploting results...")
        fig, ax = plt.subplots()
        ax.scatter(realDX, predictedDX, color="b")
        # ax.scatter(plotDX, plotPred, color="r", marker='x')
        i = 1
        while os.path.exists('ml\\logs\\sim_adv_epochs_new_bla_' + str(self.epochs) + '-Iterations_' + str(i) + '.png'):
            i += 1
        else:
            try:
                plt.savefig('ml\\logs\\sim_adv_epochs_new_bla_' + str(self.epochs) + '-Iterations_' + str(i) + '.png')
                print("saved plot as: sim_adv_epochs_new_bla_" + str(self.epochs) + '-Iterations_' + str(i) + '.png')
            except:
                print("couldnt save plot...")

    def predictTrainData(self, model, myValidDataList):
        self.createValidSet(myValidDataList)
        myValidDataList = np.array(myValidDataList)
        myValidInData = myValidDataList[:, 9:56]
        myValidOutData = myValidDataList[:, 0:2]
        #print(myValidInData)
        predictions = model.predict(myValidInData)
        #print(myValidInData)
        predictedDX = []
        realDX = []
        for i in range(len(myValidDataList)):
            predictedRaw = (int(round(predictions[i][0],0)), int(round(predictions[i][1],0)))
            print("line=%s, XY=%s, Predicted=%s, Real=%s" % (i, myValidInData[i], predictedRaw, myValidOutData[i]))
            predictedDX.append(predictions[i][0])
            realDX.append(myValidOutData[i][0])
        return predictedDX, realDX

    def predictMyData(self, model):
        # 'targetX', 'targetY', 'targetSize', 'initMouseX', 'initMouseY', 'targetID'
        myData = [
            [200, 200, 40, 100, 100, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-300, 100, 20, -200, 30, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [811,932, 12, -301, 590,3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-500, -732, 44, 32, 500, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [811, 932, 12, -301, 590, 3, 5, 3, 3, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        myData = np.array(myData)
        #print(myData)
        myPredictedData = model.predict(myData)
        for i in range(len(myData)):
            predictedRaw = (int(round(myPredictedData[i][0],0)), int(round(myPredictedData[i][1],0)))
            print("line=%s, XY=%s, Predicted=%s" % (i, myData[i], predictedRaw))

    def createValidSet(self, myValidDataList):
        for i in range(0, self.validSize):
            myTempSample = list(self.dataQueue.get())
            myTempSample.extend(self.pastDxDyList)
            self.pastDxDyList.pop(0)
            self.pastDxDyList.pop(0)
            self.pastDxDyList.append(myTempSample[0])
            self.pastDxDyList.append(myTempSample[1])
            myValidDataList.append(myTempSample)

    def createTrainSet(self, myTrainDataList):
        for i in range(0, self.trainSize):
            myTempSample = list(self.dataQueue.get())
            #print("1"+str(myTempSample))
            myTempSample.extend(self.pastDxDyList)
            #print("2"+str(myTempSample))
            self.pastDxDyList.pop(0)
            self.pastDxDyList.pop(0)
            #print(self.pastDxDyList)
            self.pastDxDyList.append(myTempSample[0])
            self.pastDxDyList.append(myTempSample[1])
            #print(self.pastDxDyList)
            myTrainDataList.append(myTempSample)
            #print(myTrainDataList)
            # print(myTempSample, len(myTempSample))