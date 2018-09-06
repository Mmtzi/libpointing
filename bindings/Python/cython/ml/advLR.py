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
            print(convInputSet.shape)
            print(sizeInputSet.shape)
            print(outDxDySet.shape)
            print(outButtonSet.shape)


            # define and fit the final model

            if os.path.exists('ml\\models/sim_lstm_20dx_20dist_sizeo.h5'):
                model = load_model('ml\\models/sim_lstm_20dx_20dist_sizeo.h5')
                print("loaded model")
            else:
                timeInput = Input(shape=(20, 4) , dtype='float32', name='convInput')
                lstm = LSTM(10)(timeInput)
                #flat = Flatten()(lstm)
                sizeInput = Input(shape=(1, ), name='sizeInput')
                x = concatenate([lstm, sizeInput])
                x = Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
                outDxDy = Dense(2, activation='linear')(x)
                outButton = Dense(1, activation='sigmoid')(x)
                model=Model([timeInput, sizeInput], [outDxDy, outButton])
                model.compile(Adam(lr=0.0005), loss=['mse', 'binary_crossentropy'])

            model.summary()
            model.fit([convInputSet, sizeInputSet], [outDxDySet, outButtonSet], epochs=self.epochs, verbose=2, batch_size=20, shuffle=True, validation_split=0.1)
            model.save('ml\\models\\sim_lstm_20dx_20dist_sizeo.h5')

            #self.predictMyData(model)
            predictedDX, realDX = self.predictTrainData(model)

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

    def predictTrainData(self, model):
        convInputSet, sizeInputSet, outDxDySet, outButtonSet = self.createValidSet()
        convInputSet = np.array(convInputSet)
        sizeInputSet = np.array(sizeInputSet)
        outDxDySet = np.array(outDxDySet)
        outButtonSet = np.array(outButtonSet)
        print(convInputSet.shape)
        print(sizeInputSet.shape)
        print(outDxDySet.shape)
        print(outButtonSet.shape)
        print(convInputSet)
        print(sizeInputSet)
        predictedDXDY, predictedButton = model.predict([convInputSet, sizeInputSet])
        #print(myValidInData)
        predictedDX = []
        realDX = []
        convInputSet = convInputSet.tolist()
        #print(convInputSet)
        sizeInputSet = sizeInputSet.tolist()
        #print(sizeInputSet)
        outDxDySet = outDxDySet.tolist()
        #print(outDxDySet)
        outButtonSet = outButtonSet.tolist()
        #print(outButtonSet)
        for i in range(0,self.validSize):
            predictedRaw = (int(round(predictedDXDY[i][0],0)), int(round(predictedDXDY[i][1],0)), int(round(predictedButton[i][0],0)))
            if i%20==0 or outButtonSet[i][0] ==1:
                print("line=%s, XY=%s, Predicted=%s, Real=%s" % (i, (convInputSet[i], sizeInputSet[i][0]), predictedRaw, (outDxDySet[i], outButtonSet[i][0])))
            predictedDX.append(predictedDXDY[i][0])
            realDX.append(outDxDySet[i][0])
        return predictedDX, realDX


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

                self.pastList.append(self.myPreTempSample[0])
                self.pastList.append(self.myPreTempSample[1])
                self.pastList.append(self.myPreTempSample[8])
                self.pastList.append(self.myPreTempSample[9])

            sizeInputList.append([myTempSample[14]])
            outDxDyList.append([myTempSample[0], myTempSample[1]])
            outButtonList.append([myTempSample[2]])
            timeSeries = np.array(self.pastList)
            timeSeries = np.reshape(timeSeries, (-1, 4))
            #print(timeSeries)
            convInputList.append(timeSeries)
            self.myPreTempSample = myTempSample

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

                self.pastList.append(self.myPreTempSample[0])
                self.pastList.append(self.myPreTempSample[1])
                self.pastList.append(self.myPreTempSample[8])
                self.pastList.append(self.myPreTempSample[9])

            sizeInputList.append([myTempSample[14]])
            outDxDyList.append([myTempSample[0], myTempSample[1]])
            outButtonList.append([myTempSample[2]])
            timeSeries = np.array(self.pastList)
            timeSeries = np.reshape(timeSeries, (-1,4))
            #print(timeSeries)
            convInputList.append(timeSeries)
            self.myPreTempSample = myTempSample

        return convInputList, sizeInputList, outDxDyList, outButtonList