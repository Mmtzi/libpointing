from keras.models import Sequential, load_model, Model
from keras.layers import Dense,BatchNormalization, Activation, Dropout, Reshape, Input, Softmax, Conv2D, Conv1D, MaxPooling2D, concatenate, Flatten, TimeDistributed
from keras.optimizers import Adam
from keras import regularizers
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import TensorBoard
from keras.callbacks import Callback, ModelCheckpoint
import tensorboard
from numpy import genfromtxt
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from thesis import plotData
from keras import backend as K
import os
import numpy as np
from threading import Thread
from queue import Queue
import time
import itertools
from random import randint
import tensorflow as tf
from multiprocessing import Pool


class Simulator(Thread):
    def __init__(self, q, trainingSet, validSet, modelname, epochs, lr, batchSize, pastTS):
        self.lr = lr
        self.modelname = modelname
        self.tbCallBack = TensorBoard(log_dir='ml\\logs\\tb/'+str(modelname), histogram_freq=0,
          write_graph=True, write_images=True)
        self.chk = ModelCheckpoint("ml\\models\\"+str(modelname), monitor='val_loss', save_best_only=False)
        self.sleepInterval = 0.2
        self.pastTimeSteps = pastTS
        self.batchSize = batchSize
        self.epochs = epochs

        self.inputNP, self.outDxDyNP, self.outButtonNP = self.prepareData(q, trainingSet)

        self.validInputNP, self. validOutDxDyNP, self.validOutButtonNP = self.prepareData(q, validSet)

        super().__init__()

    def run(self):
    # load Data
    # 'dx', 'dy', 'button', 'rx', 'ry', 'time', 'distance', 'targetID', 'directionX', 'directionY',
    # 'targetX', 'targetY', 'initMouseX', 'initMouseY', 'targetSize'

    # load or define the model

        if os.path.exists('ml\\models\\'+str(self.modelname)):
            model = load_model('ml\\models\\'+str(self.modelname))
            print("loaded model: "+str(self.modelname))
            print(K.get_value(model.optimizer.lr))
            K.set_value(model.optimizer.lr, self.lr)
        else:
            print("new model: "+ str(self.modelname))
            timeInput = Input(shape=(self.pastTimeSteps, 5), dtype='float32', name='timeInput')
            #norm = BatchNormalization()(timeInput)
            conv1 = Conv1D(512, kernel_size=3, padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.0001))(timeInput)
            norm = BatchNormalization()(conv1)
            conv2 = Conv1D(512, kernel_size=3, padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(norm)
            flat = Flatten()(conv2)
            dense1 = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(flat)
            outDxDy = Dense(2, activation='linear')(dense1)
            outButton = Dense(1, activation='sigmoid')(dense1)
            model=Model([timeInput], [outDxDy, outButton])
            model.compile(Adam(lr=self.lr), loss=['mse', 'binary_crossentropy'])

        model.summary()

        #fit_generator as long as dataqueue has enough elements, always create a new input set from which generator chooses random samples
        #while self.dataQueue.qsize() >= self.batchSize:
        #    print(self.dataQueue.qsize())

        model.fit_generator(generator=self.generator(),
                        steps_per_epoch=int(self.inputNP.shape[0]/self.batchSize),
                        epochs=self.epochs,
                        verbose=1,
                        validation_data=self.validGenerator(),
                        validation_steps=int(self.validInputNP.shape[0]/self.batchSize),
                        callbacks=[self.tbCallBack, self.chk])


        # model.fit([convInputSet, sizeInputSet], [outDxDySet, outButtonSet], epochs=self.epochs, verbose=2, batch_size=20, shuffle=True, validation_split=0.1)
        try:
            model.save('ml\\models\\'+str(self.modelname))
            print("saved model: "+str(str(self.modelname)))
        except:
            print("couldnt save model: "+str(self.modelname))

        K.clear_session()
        #model.predict_generator(self.validGenerator(self.pastTSInputList, self.sizeInputList, self.batchSize), verbose=2)

        #predictedDX, realDX = self.predictTrainData(model)

        #plotData.plotResults(predictedDX, realDX, self.epochs)

    def generator(self):
        print("generateTrainDataSets")
        self.indexList = np.arange(0, self.inputNP.shape[0], 1)
        print(self.inputNP.shape[0], self.outButtonNP.shape[0], self.outDxDyNP.shape[0], self.indexList.size)
        input = np.zeros((self.batchSize, self.pastTimeSteps, self.inputNP.shape[2]))
        outdxdy = np.zeros((self.batchSize, self.outDxDyNP.shape[1]))
        outbutton = np.zeros((self.batchSize, 1))
        pickedIndex = np.zeros((self.batchSize,1))
        print(input.shape, outdxdy.shape, outbutton.shape)
        while True:
            if self.indexList.shape[0] >= self.batchSize:
                yield self.createTrainBatch(input, outdxdy, outbutton, pickedIndex)
            else:
                self.indexList = np.arange(0, self.inputNP.shape[0], 1)

    def createTrainBatch(self, input, outdxdy, outbutton, pickedIndex):
        for i in range(0, self.batchSize):
            index = randint(0, self.indexList.shape[0]-1)
            pick = self.indexList.item(index)
            pickedIndex[i] = index
            input[i] = self.inputNP[pick]
            outdxdy[i] = self.outDxDyNP[pick]
            outbutton[i] = self.outButtonNP[pick]
            #print(i, index, pick, input[i], outdxdy[i], outbutton[i])
        self.indexList = np.delete(self.indexList, pickedIndex)
        return [input], [outdxdy, outbutton]

    def validGenerator(self):
        print("generateValidDataSets")
        self.validIndexList = np.arange(0, self.validInputNP.shape[0], 1)
        print(self.validInputNP.shape[0], self.validOutButtonNP.shape[0], self.validOutDxDyNP.shape[0], self.validIndexList.size)
        input = np.zeros((self.batchSize, self.pastTimeSteps, self.validInputNP.shape[2]))
        outdxdy = np.zeros((self.batchSize, self.validOutDxDyNP.shape[1]))
        outbutton = np.zeros((self.batchSize, 1))
        pickedIndex = np.zeros((self.batchSize, 1))
        print(input.shape, outdxdy.shape, outbutton.shape)
        while True:
            if self.validIndexList.shape[0] >= self.batchSize:
                yield self.createValidBatch(input, outdxdy, outbutton, pickedIndex)
            else:
                self.validIndexList = np.arange(0, self.validInputNP.shape[0], 1)

    def createValidBatch(self, input, outdxdy, outbutton, pickedIndex):
        for i in range(0, self.batchSize):
            index = randint(0, self.validIndexList.shape[0]-1)
            pick = self.validIndexList.item(index)
            pickedIndex[i] = pick
            input[i] = self.validInputNP[pick]
            outdxdy[i] = self.validOutDxDyNP[pick]
            outbutton[i] = self.validOutButtonNP[pick]
        self.validIndexList = np.delete(self.validIndexList, pickedIndex)
        return [input], [outdxdy, outbutton]

    def prepareData(self, q, dataSet):
        print("preparing Data...")
        print(dataSet.shape[0])
        input=[]
        outdxdy= []
        outbutton =[]
        clicks = 0
        noclicks =0
        if dataSet.shape[0] == 0:
            dataSet = np.array(list(q.queue))
        dataSet = dataSet[:, [0, 1, 2, 7, 8, 9, 14]]
        for i in range(self.pastTimeSteps, dataSet.shape[0]):
            batchsample = dataSet[i - self.pastTimeSteps: i + 1, :]
            #print(batchsample)
            if ((98 in batchsample[:, 3] or 99 in batchsample[:, 3] or 100 in batchsample[:, 3]) and 1 in batchsample[:, 3]):
                #print(batchsample)
                continue
            else:
                inputSlice = batchsample[:self.pastTimeSteps, [0,1,4,5,6]]
                outdxdySlice = batchsample[self.pastTimeSteps, [0,1]]
                outbuttonSlice = batchsample[self.pastTimeSteps, 2]
                input.append(inputSlice)
                outdxdy.append(outdxdySlice)
                outbutton.append(outbuttonSlice)
                if int(outbuttonSlice) == 1:
                    clicks += 1
                    for k in range(0, 20):
                        input.append(inputSlice)
                        outdxdy.append(outdxdySlice)
                        outbutton.append(outbuttonSlice)
                        clicks += 1
                else:
                    noclicks += 1
        print(clicks/(noclicks+clicks))
        return np.array(input), np.array(outdxdy), np.array(outbutton)

