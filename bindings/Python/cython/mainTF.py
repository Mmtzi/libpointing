import _thread
import myFitsLawGame, SimulatorTest
from ml import advLR, recurrentSim, myActorCritic, calcScore, environment, ActorCritic, testActorCritic
from queue import Queue
from numpy import genfromtxt
from glob import glob
from random import shuffle
import os
import numpy as np
import csv
import pandas as pd
#import itertools,collections
import time
from tensorflow.python.client import device_lib
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def main():
    epochs = 20
    lr = 0.0001
    batchsize = 128
    pastTS = 30
    modelexp = "512dr0001_512dr0001_32dr0001"
    specials = "inDistanceAug10nAdv"
    modelname=str(epochs)+"e_"+str(lr)+"lr_"+str(batchsize)+"b_"+str(pastTS)+"TS_"+str(modelexp)+"_"+str(specials)+".h5"
    #dataQueue with sample data from fitsLawGame, every Thread has Acces to
    simuQueue = Queue()
    actorQueueUser = Queue()
    actorQueueSimu = Queue()
    #allTrainData = glob('thesis/logData/adaptive/system*.csv')
    #allValidData = glob('thesis/logData/valid/system*.csv')
    #updateData(allTrainData)
    #updateData(allValidData)
    allTrainData = glob('thesis/logData/adaptive/system*.csv')
    #allTrainData = []
    allValidData = glob('thesis/logData/valid/system*.csv')
    onTheFly = False
    gameThread = collectData(simuQueue, actorQueueUser)
    #trainSimThread = trainSimulator(simuQueue,  modelname, onTheFly, epochs, lr, batchsize, pastTS, allTrainData, allValidData)
    #testSimThread = testSimulator(actorQueueSimu, modelname, pastTS)
    actorCriticThread = trainActor(actorQueueUser, actorQueueSimu)

def collectData(dataQueue, actorQueue):
    try:
        print("trying to init Game thread")
        gameThread = myFitsLawGame.Game(dataQueue, actorQueue)
        print("trying to start Game thread")
        gameThread.start()
        print("Game thread started")
        return gameThread

    except:
        print("unable to start Gaming thread")


def updateData(allTrainData):
    for each in allTrainData:
        if os.path.getsize(each) > 300000:
            file = open(each, 'r')
            r = csv.reader(file)
            row0 = next(r)
            print(row0)
            row0.append('score')
            allSamples =[]
            allSamples.append(row0)
            first = True
            for sample in r:
                if first:
                    oldsample = sample
                    first = False
                sample.append(calcScore.calcScoreOfAction((float(sample[10])-float(sample[8]), float(sample[11])-float(sample[9])),
                                                          (float(oldsample[10])-float(oldsample[8]), float(oldsample[11])-float(oldsample[9])),
                                                          float(sample[6]), float(oldsample[6]), (float(sample[10]), float(sample[11])),
                                                          float(sample[14]), float(sample[5]), float(sample[2])))
                oldsample = sample
                allSamples.append(sample)
            print(allSamples)
            file.close()
            filew = open(each, "w", newline='')
            w = csv.writer(filew)
            w.writerows(allSamples)
            filew.close()
            print("Added Score to File:" + str(each))

def trainSimulator(dataQueue, modelname, onTheFly, epochs, lr, batchSize, pastTS, allTrainData, allValidData):
    trainingSet = np.empty(0)
    validSet = []
    first = True
    for each in allValidData:
        if os.path.getsize(each) > 300000:
            loadedVSet = genfromtxt(each, delimiter=',', skip_header=1)
            print(each, loadedVSet.shape)
            if loadedVSet.shape[1] > 15:
                loadedVSet = loadedVSet[:, :-1]
            if first == True:
                validSet = loadedVSet
                first = False
            else:
                validSet = np.append(validSet, loadedVSet, axis=0)
    if onTheFly:
        print("trying to init on the fly TrainSimulator thread")
        simulatorThread = advLR.Simulator(dataQueue, trainingSet, validSet, modelname, epochs, lr, batchSize, pastTS)
        print("trying to start on the fly TrainSimulator thread")
        simulatorThread.start()
        print("on the fly TrainSimulator thread started")
    else:
        sorted(allTrainData, key=os.path.getmtime)
        first = True
        for each in allTrainData:
            if os.path.getsize(each) > 300000:
                loadedTSet = genfromtxt(each, delimiter=',', skip_header=1)
                print(each, loadedTSet.shape)
                if loadedTSet.shape[1] > 15:
                    loadedTSet = loadedTSet[:,:-1]
                if first == True:
                    trainingSet = loadedTSet
                    first = False
                else:
                    trainingSet = np.append(trainingSet, loadedTSet, axis=0)
        try:
            print("trying to init TrainSimulator thread")
            simulatorThread = advLR.Simulator(dataQueue, trainingSet, validSet, modelname, epochs, lr, batchSize,pastTS)
            print("trying to start TrainSimulator thread")
            simulatorThread.start()
            print("TrainSimulator thread started")
        except:
            print("unable to start TrainSimulator thread")

def testSimulator(actorQueue, modelname, pastTimeStepsSimulator):
    try:
        print("trying to init SimTest thread")
        simTestThread = SimulatorTest.SimTest(actorQueue, modelname, pastTimeStepsSimulator)
        print("trying to start SimTest thread")
        simTestThread.start()
        print("SimTest thread started")
    except:
        print("unable to start SimTest thread")

def trainActor(actorQueueUser, actorQueueSimu):
    print("trying to init env thread")
    actCritic = ActorCritic.ActorCritic(actorQueueUser, actorQueueSimu)
    print("trying to start env thread")
    actCritic.start()
    print("trainActor env started")
        #print("unable to start trainActor thread")

if __name__ == '__main__':
    main()