from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Input, Softmax, Conv2D, Conv1D, MaxPooling2D, concatenate, Flatten
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
        self.pastList = []
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
            # 'targetX', 'targetY', 'initMouseX', 'initMouseY', 'targetSize'

            myValidDataList = []
            sleepTime = 0
            print(self.dataQueue.qsize())

            while self.dataQueue.qsize() <= (self.trainSize +self.validSize):
                time.sleep(self.sleepInterval)
                sleepTime +=self.sleepInterval


            convInputSet, sizeInputSet, outDxDySet, outButtonSet = self.createInputSet()
            convInputSet = np.array(convInputSet)
            sizeInputSet = np.array(sizeInputSet)
            outDxDySet = np.array(outDxDySet)
            outButtonSet = np.array(outButtonSet)
            print(convInputSet.shape, convInputSet)
            print(sizeInputSet.shape)
            print(outDxDySet.shape, outDxDySet)
            print(outButtonSet.shape)


            # define and fit the final model

            if os.path.exists('ml\\models/sim_conv_20dx_20dist_sizeo.h5'):
                model = load_model('ml\\models/sim_conv_20dx_20dist_sizeo.h5')
                print("loaded model")
            else:
                convInput = Input(shape=(900, 20, 4) , dtype='float32', name='convInput')
                conv1 = Conv2D(kernel_size=(1, 4), data_format="channels_last", strides=1, filters=1 , activation='relu')(convInput)
                max2 = MaxPooling2D(pool_size=(1,1))(conv1)
                flat = Flatten()(max2)
                print(flat)
                sizeInput = Input(shape=(20, ), name='sizeInput')
                x = concatenate([flat, sizeInput], axis=1)
                x = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
                outDxDy = Dense(2, activation='linear')(x)
                outButton = Dense(1, activation='sigmoid')(x)
                model=Model([convInput, sizeInput], [outDxDy, outButton])
                model.compile(Adam(lr=0.0005), loss=['mse', 'binary_crossentropy'])

            model.summary()
            model.fit([convInputSet, sizeInputSet], [outDxDySet, outButtonSet], epochs=self.epochs, verbose=2, batch_size=20, shuffle=True, validation_split=0.1)
            model.save('ml\\models\\sim_conv_20dx_20dist_sizeo.h5')

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
        convInputSet, sizeInputSet, outDxDySet, outButtonSet = self.createValidSet()
        convInputSet = np.array(convInputSet)
        sizeInputSet = np.array(sizeInputSet)
        outDxDySet = np.array(outDxDySet)
        outButtonSet = np.array(outButtonSet)
        print(convInputSet.shape, convInputSet)
        print(sizeInputSet.shape)
        print(outDxDySet.shape, outDxDySet)
        print(outButtonSet.shape)
        predictedDXDY, predictedButton = model.predict([convInputSet, sizeInputSet])
        #print(myValidInData)
        predictedDX = []
        realDX = []
        k = 0
        for i in range(len(myValidDataList)):
            predictedRaw = (int(round(predictedDXDY[i][0],0)), int(round(predictedDXDY[i][1],0)), int(round(predictedButton[i][0],0)))
            if i%20==0 or outButtonSet[0][i] ==1:
                print("line=%s, XY=%s, Predicted=%s, Real=%s" % (i, (convInputSet[i], sizeInputSet[i][0]), (predictedRaw[i]), (outDxDySet[i][0], outButtonSet[i][1])))
            predictedDX.append(predictedDXDY[i][0])
            realDX.append(outDxDySet[i][0])
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

    def createValidSet(self):
        convInputList = []
        sizeInputList = []
        outDxDyList = []
        outButtonList = []
        for i in range(0, self.validSize):
            myTempSample = list(self.dataQueue.get())
            if len(self.pastList) == 0:
                self.pastList = [0, 0, myTempSample[8], myTempSample[9]] * self.pastTimeSteps
            else:
                self.pastList.pop(0)
                self.pastList.pop(0)
                self.pastList.pop(0)
                self.pastList.pop(0)

                self.pastList.append(myPreTempSample[0])
                self.pastList.append(myPreTempSample[1])
                self.pastList.append(myPreTempSample[8])
                self.pastList.append(myPreTempSample[9])

            sizeInputList.append([myTempSample[14]])
            outDxDyList.append([myTempSample[0], myTempSample[1]])
            outButtonList.append([myTempSample[2]])
            timeSeries = np.array(self.pastList)
            timeSeries = np.reshape(timeSeries, (-1, 4))
            print(timeSeries)
            convInputList.append(timeSeries)
            myPreTempSample = myTempSample

        return convInputList, sizeInputList, outDxDyList, outButtonList

    def createInputSet(self):
        convInputList = []
        sizeInputList = []
        outDxDyList = []
        outButtonList = []
        for i in range(0, self.trainSize):
            myTempSample = list(self.dataQueue.get())
            if len(self.pastList) == 0:
                self.pastList = [0 , 0, myTempSample[8], myTempSample[9]]* self.pastTimeSteps
            else:
                self.pastList.pop(0)
                self.pastList.pop(0)
                self.pastList.pop(0)
                self.pastList.pop(0)

                self.pastList.append(myPreTempSample[0])
                self.pastList.append(myPreTempSample[1])
                self.pastList.append(myPreTempSample[8])
                self.pastList.append(myPreTempSample[9])

            sizeInputList.append([myTempSample[14]])
            outDxDyList.append([myTempSample[0], myTempSample[1]])
            outButtonList.append([myTempSample[2]])
            timeSeries = np.array(self.pastList)
            timeSeries = np.reshape(timeSeries, (-1,4))
            print(timeSeries)
            convInputList.append(timeSeries)
            myPreTempSample = myTempSample

        return convInputList, sizeInputList, outDxDyList, outButtonList