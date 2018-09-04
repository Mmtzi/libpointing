from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Input, Softmax, Conv1D
from keras.optimizers import Adam
from keras import regularizers
from keras.layers.recurrent import LSTM, GRU
from numpy import genfromtxt
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from thesis import plotData
import os
import numpy as np
from threading import Thread
from queue import Queue
import time


class Simulator(Thread):
    def __init__(self, q):
        self.epochs = 200
        self.dataQueue = q
        self.sleepInterval = 0.2
        self.pastTimeSteps = 20
        self.trainSize = 900
        self.validSize = 100
        self.pastDxDyList = [0]*self.pastTimeSteps*2
        self.pastDistanceList = []
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
            # 'dx', 'dy', 'button', 'rx', 'ry', 'time', 'distance', 'targetID', 'directionX', 'directionY',
            # 'targetX', 'targetY', 'targetSize', 'initMouseX', 'initMouseY',

            myTrainDataList = []
            myValidDataList = []
            sleepTime = 0
            print(self.dataQueue.qsize())

            while self.dataQueue.qsize() <= (self.trainSize +self.validSize):
                time.sleep(self.sleepInterval)
                sleepTime +=self.sleepInterval


            self.createTrainSet(myTrainDataList)

            myTrainDataList = np.array(myTrainDataList)
            myTrainInData = myTrainDataList[:, 14:95]
            myTrainOutData = myTrainDataList[:,0:3]
            labelDxDy = myTrainOutData[:,0:2]
            labelButton = myTrainOutData[:,2:3]

            # define and fit the final model

            if os.path.exists('ml\\models/sim_par_20dx_20dist_sizeo.h5'):
                model = load_model('ml\\models/sim_par_20dx_20dist_sizeo.h5')
                print("loaded model")
            else:
                main_input = Input(shape=(81,), dtype='float32', name='main_input')
                x = Co
                x = Dense(42, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(main_input)
                x = Dense(9, activation='relu')(x)
                outDxDy = Dense(2, activation='linear')(x)
                outButton = Dense(1, activation='sigmoid')(x)
                model=Model(main_input, [outDxDy, outButton])
                model.compile(Adam(lr=0.0005), loss=['mse', 'binary_crossentropy'])

            model.summary()
            model.fit(myTrainInData, [labelDxDy, labelButton], epochs=self.epochs, verbose=2, batch_size=20, shuffle=True, validation_split=0.1)
            model.save('ml\\models\\sim_par_20dx_20dist_sizeo.h5')

            #self.predictMyData(model)

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

            plotData.plotResults(predictedDX, realDX, self.epochs)

    def predictTrainData(self, model, myValidDataList):
        self.createValidSet(myValidDataList)
        myValidDataList = np.array(myValidDataList)
        myValidInData = myValidDataList[:, 14:95]
        myValidOutData = myValidDataList[:, 0:3]
        validDxDy = myValidOutData[:, 0:2]
        myValidButton = myValidOutData[:, 2:3]
        #print(myValidInData)
        predictedDXDY, predictedButton = model.predict(myValidInData)
        #print(myValidInData)
        predictedDX = []
        realDX = []
        k = 0
        for i in range(len(myValidDataList)):
            predictedRaw = (int(round(predictedDXDY[i][0],0)), int(round(predictedDXDY[i][1],0)), int(round(predictedButton[i][0],0)))
            if i%20==0 or myValidButton[i] ==1:
                print("line=%s, XY=%s, Predicted=%s, Real=%s" % (i, myValidInData[i], predictedRaw, (validDxDy[i], myValidButton[i])))
            predictedDX.append(predictedDXDY[i][0])
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
        for i in range(0, self.validSize):
            myTempSample = list(self.dataQueue.get())
            if len(self.pastDistanceList) == 0:
                self.pastDistanceList = [myTempSample[8], myTempSample[9]]* self.pastTimeSteps
            myTempSample.extend(self.pastDxDyList + self.pastDistanceList)

            self.pastDxDyList.pop(0)
            self.pastDxDyList.pop(0)
            self.pastDistanceList.pop(0)
            self.pastDistanceList.pop(0)

            self.pastDxDyList.append(myTempSample[0])
            self.pastDxDyList.append(myTempSample[1])
            self.pastDistanceList.append(myTempSample[8])
            self.pastDistanceList.append(myTempSample[9])

            myValidDataList.append(myTempSample)

    def createTrainSet(self, myTrainDataList):
        for i in range(0, self.trainSize):
            myTempSample = list(self.dataQueue.get())
            if len(self.pastDistanceList) == 0:
                self.pastDistanceList = [myTempSample[8], myTempSample[9]]* self.pastTimeSteps
                print(self.pastDistanceList)
            myTempSample.extend(self.pastDxDyList + self.pastDistanceList)

            self.pastDxDyList.pop(0)
            self.pastDxDyList.pop(0)
            self.pastDistanceList.pop(0)
            self.pastDistanceList.pop(0)

            self.pastDxDyList.append(myTempSample[0])
            self.pastDxDyList.append(myTempSample[1])
            self.pastDistanceList.append(myTempSample[8])
            self.pastDistanceList.append(myTempSample[9])

            #print(myTempSample)

            myTrainDataList.append(myTempSample)

            # print(myTempSample, len(myTempSample))