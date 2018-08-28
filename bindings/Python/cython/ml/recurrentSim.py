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
        self.epochs = 100
        self.dataQueue = q
        self.sleepInterval = 0.2
        self.pastTimeSteps = 20
        self.trainSize = 1
        self.validSize = 1
        self.look_back = 10
        self.myDataSample = []
        super().__init__()


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

            myTrainInData, myTrainOutData = self.createTrainSet()

            #print(myTrainInData)
            myTrainInData = np.array(myTrainInData)
            myTrainOutData = np.array(myTrainOutData)
            #myTrainInData = np.reshape(myTrainInData, (myTrainInData.shape[0], myTrainInData.shape[1], 1))
            print(np.shape(myTrainInData))
            #print(myTrainOutData)
            print(np.shape(myTrainOutData))
            # define and fit the final model

            if os.path.exists('ml\\models/sim_adv_LSTM_model.h5'):
                model = load_model('ml\\models/sim_adv_LSTM_model.h5')
                print("loaded model")
            else:
                model = Sequential()
                model.add(LSTM(16, input_shape=(self.look_back, 6)))
                model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
                model.add(Dense(2, activation='linear'))
                model.compile(Adam(lr=0.001), loss='mse')

            model.fit(myTrainInData, myTrainOutData, epochs=self.epochs, verbose=2, batch_size=10, shuffle=False)
            model.save('ml\\models\\sim_adv_LSTM_model.h5')

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
                plt.savefig('ml\\logs\\sim_adv_epochs_LSTM_' + str(self.epochs) + '-Iterations_' + str(i) + '.png')
                print("saved plot as: sim_adv_epochs_LSTM_" + str(self.epochs) + '-Iterations_' + str(i) + '.png')
            except:
                print("couldnt save plot...")

    def predictTrainData(self, model, myValidDataList):
        self.createValidSet(myValidDataList)
        myValidDataList = np.array(myValidDataList)
        myValidInData = myValidDataList[:, 9:15]
        myValidOutData = myValidDataList[:, 0:2]
        myValidInData = np.reshape(myValidInData, (myValidInData.shape[0], self.look_back, myValidInData.shape[1]))
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

        #myData = np.array(myData)
        #print(myData)
        #myPredictedData = model.predict(myData)
        #for i in range(len(myData)):
            #predictedRaw = (int(round(myPredictedData[i][0],0)), int(round(myPredictedData[i][1],0)))
            #print("line=%s, XY=%s, Predicted=%s" % (i, myData[i], predictedRaw))

    def createValidSet(self, myValidDataList):
        for i in range(0, self.trainSize):
            myTempSample = list(self.dataQueue.get())
            myValidDataList.append(myTempSample)
            while len(myValidDataList) <= self.look_back:
                    myValidDataList.append(myTempSample)
        return myValidDataList

    def createTrainSet(self):
        myTrainDataList = []
        myLabelDataList = []
        for i in range(0, self.trainSize):
            myTempSample = list(self.dataQueue.get())
            self.myDataSample.append(myTempSample)
            while len(self.myDataSample) < self.look_back:
                    self.myDataSample.append(myTempSample)
            self.myDataNumpySample = np.array(self.myDataSample)
            myTrainDataList.append(self.myDataNumpySample[:, 9:15])
            myLabelDataList.append(self.myDataNumpySample[:, 0:2])
            self.myDataSample.pop(0)
            print(self.myDataSample)
        #print(myTrainDataList)
        return myTrainDataList, myLabelDataList