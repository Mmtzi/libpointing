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
import itertools
from random import randint


class Simulator(Thread):
    def __init__(self, q):
        self.dataQueue = q
        self.sleepInterval = 0.2
        self.pastTimeSteps = 20
        self.batchSize = 128
        #past dx dy distX distY list length: pastTimeSteps*4
        self.pastTempList = []
        #inputLists
        self.pastTSInputList = [] #x*20*4
        self.sizeInputList = [] #x*1
        #labelLists
        self.labelDxDyList = [] #x*2
        self.labelButtonList = [] #x*1
        super().__init__()

    def run(self):
        # load Data
        # 'dx', 'dy', 'button', 'rx', 'ry', 'time', 'distance', 'targetID', 'directionX', 'directionY',
        # 'targetX', 'targetY', 'initMouseX', 'initMouseY', 'targetSize'
        sleepTime = 0

        #wait till dataqsize > batchsize
        while self.dataQueue.qsize() <= (self.batchSize):
            time.sleep(self.sleepInterval)
            sleepTime +=self.sleepInterval

        # load or define the model

        if os.path.exists('ml\\models/sim_lstm_fitg_20dx_20dist_sizeo.h5'):
            model = load_model('ml\\models/sim_lstm_fitg_20dx_20dist_sizeo.h5')
            print("loaded model")
        else:
            timeInput = Input(shape=(20, 4) , dtype='float32', name='timeInput')
            lstm = LSTM(20)(timeInput)
            #flat = Flatten()(lstm)
            sizeInput = Input(shape=(1,), name='sizeInput')
            x = concatenate([lstm, sizeInput])
            x = Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
            outDxDy = Dense(2, activation='linear')(x)
            outButton = Dense(1, activation='sigmoid')(x)
            model=Model([timeInput, sizeInput], [outDxDy, outButton])
            model.compile(Adam(lr=0.0005), loss=['mse', 'binary_crossentropy'])

        model.summary()

        #fit_generator as long as dataqueue has enough elements, always create a new input set from which generator chooses random samples
        #while self.dataQueue.qsize() >= self.batchSize:
        #    print(self.dataQueue.qsize())
        model.fit_generator(self.generator(),
                            steps_per_epoch=500000/self.batchSize,
                            nb_epoch=50, verbose=1)


        # model.fit([convInputSet, sizeInputSet], [outDxDySet, outButtonSet], epochs=self.epochs, verbose=2, batch_size=20, shuffle=True, validation_split=0.1)
        model.save('ml\\models\\sim_lstm_fitg_20dx_20dist_sizeo.h5')

        #model.predict_generator(self.validGenerator(self.pastTSInputList, self.sizeInputList, self.batchSize), verbose=2)

        #predictedDX, realDX = self.predictTrainData(model)

        #plotData.plotResults(predictedDX, realDX, self.epochs)

    def generator(self):
        while True:
            #print("generateDataSet")
            pastTSInputList = []
            batch_pointSize = []
            batch_dxdy = []
            batch_button = []
            l = list(self.dataQueue.queue)
            for i in range(self.batchSize):
                start = randint(0, len(l) - self.pastTimeSteps -1)
                batchsample = np.array(l[start : start + self.pastTimeSteps + 1])
                #print(batchsample.shape)

                batch_pointSize.append(batchsample[20, 14])
                #print("sizeShape:"+str(np.array(batch_pointSize).shape))

                batch_button.append(batchsample[20, 2])
                #print("buttonShape:"+str(np.array(batch_button).shape))

                pastTSInputList.append(batchsample[:20, [0,1,8,9]])
                #print("pastTSSape:"+str(np.array(pastTSInputList).shape))
                #print(batchsample[:20, [0,1,8,9]], batchsample[20, [0,1]])
                batch_dxdy.append(batchsample[20, [0,1]])
                #print("dxdyShappe:"+str(np.array(batch_dxdy).shape))

            yield [np.array(pastTSInputList), np.array(batch_pointSize)], [np.array(batch_dxdy), np.array(batch_button)]


    def generator2(self):
        self.createInputSet()
        pastTSInputList = self.pastTSInputList
        sizeInputList = self.sizeInputList
        labelDxDyList = self.labelDxDyList
        labelButtonList = self.labelButtonList
        batch_size = self.batchSize
        #Create empty arrays to contain batch of features and labels

        batch_pastTS = np.zeros((batch_size, 20, 4))
        batch_pointSize = np.zeros((batch_size, 1))
        batch_dxdy = np.zeros((batch_size, 2))
        batch_button = np.zeros((batch_size, 1))
        while True:
            for i in range(0, batch_size-1):
                    # choose random index in input
                index = np.random.choice(len(pastTSInputList), 1)
                batch_pastTS[i] = np.array(pastTSInputList)[index][0]
                batch_pointSize[i] = np.array(sizeInputList)[index][0]
                batch_dxdy[i] = np.array(labelDxDyList)[index][0]
                batch_button[i] = np.array(labelButtonList)[index][0]
            yield [pastTSInputList, batch_pointSize], [batch_dxdy, batch_button]

    def createInputSet(self):
        #get batchsize samples
        for i in range(0, self.batchSize):
            myTempSample = list(self.dataQueue.get())
            #if pasttemplist is empty: init
            if len(self.pastTempList) == 0:
                self.pastTempList = [0 , 0, myTempSample[8], myTempSample[9]] * self.pastTimeSteps
            #pop "last" 4 elements and add the 4 from the sample before
            else:
                self.pastTempList.pop(0)
                self.pastTempList.pop(0)
                self.pastTempList.pop(0)
                self.pastTempList.pop(0)

                self.pastTempList.append(self.myPreTempSample[0])
                self.pastTempList.append(self.myPreTempSample[1])
                self.pastTempList.append(self.myPreTempSample[8])
                self.pastTempList.append(self.myPreTempSample[9])

            #fill the input lists with elements from act sample
            self.sizeInputList.append([myTempSample[14]])
            self.labelDxDyList.append([myTempSample[0], myTempSample[1]])
            self.labelButtonList.append([myTempSample[2]])

            #convert and reshape to 20x4
            timeSeries = np.array(self.pastTempList)
            timeSeries = np.reshape(timeSeries, (-1,4))
            #timeSeries.tolist()
            #print(timeSeries)

            #add timeseries to input list
            self.pastTSInputList.append(timeSeries)
            #remember sample for next ts
            self.myPreTempSample = myTempSample





    # TODO not done jet
    def validGenerator(self, pastTSInputList, sizeInputList, batch_size):

        # Create empty arrays to contain batch of features and labels#

        batch_pastTS = np.zeros((batch_size, 20, 4))
        batch_pointSize = np.zeros((batch_size, 1))

        while (self.dataQueue.qsize() >= batch_size):
            print(len(pastTSInputList))
            for i in range(0, batch_size - 1):
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
        # print(convInputSet)
        # print(sizeInputSet)
        predictedDXDY, predictedButton = model.predict([convInputSet, sizeInputSet])
        # print(myValidInData)
        predictedDX = []
        realDX = []
        convInputSet = convInputSet.tolist()
        # print(convInputSet)
        sizeInputSet = sizeInputSet.tolist()
        # print(sizeInputSet)
        outDxDySet = outDxDySet.tolist()
        # print(outDxDySet)
        outButtonSet = outButtonSet.tolist()
        # print(outButtonSet)

        for i in range(0, self.validSize):
            predictedRaw = (int(round(predictedDXDY[i][0], 0)), int(round(predictedDXDY[i][1], 0)),
                            int(round(predictedButton[i][0], 0)))
            if i % 20 == 0 or outButtonSet[i][0] == 1:
                print("line=%s, XY=%s, Predicted=%s, Real=%s" % (
                i, (convInputSet[i], sizeInputSet[i][0]), predictedRaw, (outDxDySet[i], outButtonSet[i][0])))
            predictedDX.append(predictedDXDY[i][0])
            realDX.append(outDxDySet[i][0])
        return predictedDX, realDX