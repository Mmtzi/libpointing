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
        #todo implement fit generator
        self.epochs = 200
        self.dataQueue = q
        self.sleepInterval = 0.2
        self.pastTimeSteps = 20
        self.batchSize = 8
        self.pastTempList = []

        self.pastTSInputList = []
        self.sizeInputList = []
        self.labelDxDyList = []
        self.labelButtonList = []
        super().__init__()

    def run(self):
        # load Data
        # 'dx', 'dy', 'button', 'rx', 'ry', 'time', 'distance', 'targetID', 'directionX', 'directionY',
        # 'targetX', 'targetY', 'initMouseX', 'initMouseY', 'targetSize'
        sleepTime = 0

        while self.dataQueue.qsize() <= (self.batchSize):
            time.sleep(self.sleepInterval)
            sleepTime +=self.sleepInterval


        self.createInputSet()

        # define and fit the final model

        if os.path.exists('ml\\models/sim_dense_fitg_20dx_20dist_sizeo.h5'):
            model = load_model('ml\\models/sim_dense_fitg_20dx_20dist_sizeo.h5')
            print("loaded model")
        else:
            timeInput = Input(shape=(20, 4) , dtype='float32', name='timeInput')
            dense = Dense(40)(timeInput)
            flat = Flatten()(dense)
            sizeInput = Input(shape=(1,), name='sizeInput')
            x = concatenate([flat, sizeInput])
            x = Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
            outDxDy = Dense(2, activation='linear')(x)
            outButton = Dense(1, activation='sigmoid')(x)
            model=Model([timeInput, sizeInput], [outDxDy, outButton])
            model.compile(Adam(lr=0.0005), loss=['mse', 'binary_crossentropy'])

        model.summary()

        while self.dataQueue.qsize() >= self.batchSize:
            print(self.dataQueue.qsize())
            model.fit_generator(self.generator(self.pastTSInputList, self.sizeInputList, self.labelDxDyList, self.labelButtonList, self.batchSize), steps_per_epoch=int(len(self.pastTSInputList)/self.batchSize), epochs=5, verbose=2)
            self.createInputSet()

        # model.fit([convInputSet, sizeInputSet], [outDxDySet, outButtonSet], epochs=self.epochs, verbose=2, batch_size=20, shuffle=True, validation_split=0.1)
        model.save('ml\\models\\sim_dense_fitg_20dx_20dist_sizeo.h5')

        #model.predict_generator(self.validGenerator(self.pastTSInputList, self.sizeInputList, self.batchSize), verbose=2)

        #self.predictMyData(model)
        #predictedDX, realDX = self.predictTrainData(model)

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

        #plotData.plotResults(predictedDX, realDX, self.epochs)

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
        #print(convInputSet)
        #print(sizeInputSet)
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

    def generator(self, pastTSInputList, sizeInputList, labelDxDyList, labelButtonList, batch_size):

        # Create empty arrays to contain batch of features and labels#

        batch_pastTS = np.zeros((batch_size, 20, 4))
        batch_pointSize = np.zeros((batch_size, 1))
        batch_dxdy = np.zeros((batch_size, 2))
        batch_button = np.zeros((batch_size, 1))
        while True:
            for i in range(0, batch_size-1):
                    # choose random index in features
                index = np.random.choice(len(pastTSInputList), 1)
                batch_pastTS[i] = np.array(pastTSInputList)[index][0]
                batch_pointSize[i] = np.array(sizeInputList)[index][0]
                batch_dxdy[i] = np.array(labelDxDyList)[index][0]
                batch_button[i] = np.array(labelButtonList)[index][0]
            yield [batch_pastTS, batch_pointSize], [batch_dxdy, batch_button]

    def validGenerator(self, pastTSInputList, sizeInputList, batch_size):

        # Create empty arrays to contain batch of features and labels#

        batch_pastTS = np.zeros((batch_size, 20, 4))
        batch_pointSize = np.zeros((batch_size, 1))

        while (self.dataQueue.qsize() >= batch_size):
            print(len(pastTSInputList))
            for i in range(0, batch_size-1):
                # choose random index in features
                index = np.random.choice(len(pastTSInputList), 1)
                batch_pastTS[i] = np.array(pastTSInputList)[index][0]
                batch_pointSize[i] = np.array(sizeInputList)[index][0]
            yield [batch_pastTS, batch_pointSize]


    def createValidSet(self):
        for i in range(0, self.batchSize):

            myTempSample = list(self.dataQueue.get())
            if len(self.pastTempList) == 0:
                self.pastTempList = [0, 0, myTempSample[8], myTempSample[9]] * self.pastTimeSteps
            else:
                self.pastTempList.pop(0)
                self.pastTempList.pop(0)
                self.pastTempList.pop(0)
                self.pastTempList.pop(0)

                self.pastTempList.append(self.myPreTempSample[0])
                self.pastTempList.append(self.myPreTempSample[1])
                self.pastTempList.append(self.myPreTempSample[8])
                self.pastTempList.append(self.myPreTempSample[9])

            self.sizeInputList.append([myTempSample[14]])
            self.labelDxDyList.append([myTempSample[0], myTempSample[1]])
            self.labelButtonList.append([myTempSample[2]])

            timeSeries = np.array(self.pastTempList)
            timeSeries = np.reshape(timeSeries, (-1, 4))

            #print(timeSeries)
            self.pastTSInputList.append(timeSeries)
            self.myPreTempSample = myTempSample

    def createInputSet(self):

        for i in range(0, self.batchSize):

            myTempSample = list(self.dataQueue.get())
            if len(self.pastTempList) == 0:
                self.pastTempList = [0 , 0, myTempSample[8], myTempSample[9]] * self.pastTimeSteps
            else:
                self.pastTempList.pop(0)
                self.pastTempList.pop(0)
                self.pastTempList.pop(0)
                self.pastTempList.pop(0)

                self.pastTempList.append(self.myPreTempSample[0])
                self.pastTempList.append(self.myPreTempSample[1])
                self.pastTempList.append(self.myPreTempSample[8])
                self.pastTempList.append(self.myPreTempSample[9])

            self.sizeInputList.append([myTempSample[14]])
            self.labelDxDyList.append([myTempSample[0], myTempSample[1]])
            self.labelButtonList.append([myTempSample[2]])

            timeSeries = np.array(self.pastTempList)
            timeSeries = np.reshape(timeSeries, (-1,4))
            timeSeries.tolist()
            #print(timeSeries)

            self.pastTSInputList.append(timeSeries)
            self.myPreTempSample = myTempSample