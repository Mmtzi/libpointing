import _thread
import myFitsLawGame, SimulatorTest
from ml import advLR, recurrentSim, myActorCritic
from queue import Queue
from numpy import genfromtxt
from glob import glob
from random import shuffle
import os
import numpy as np
#import itertools,collections
import time
from tensorflow.python.client import device_lib
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def main():
    epochs = 30
    lr = 0.0001
    batchsize = 64
    pastTS = 20
    modelexp = "512conv13_drop25_512conv13_drop25_32denser0001"
    specials = "inDistanceAug20nAdv"
    modelname=str(epochs)+"e_"+str(lr)+"lr_"+str(batchsize)+"b_"+str(pastTS)+"TS_"+str(modelexp)+"_"+str(specials)+".h5"
    #dataQueue with sample data from fitsLawGame, every Thread has Acces to
    simuQueue = Queue()
    actorQueueUser = Queue()
    actorQueueSimu = Queue()
    onTheFly = False
    trainSimThread = trainSimulator(simuQueue,  modelname, onTheFly, epochs, lr, batchsize, pastTS)
    #gameThread = collectData(simuQueue, actorQueueUser)
    #testSimThread = testSimulator(actorQueueSimu, modelname, pastTS)

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



def trainSimulator(dataQueue, modelname, onTheFly, epochs, lr, batchSize, pastTS):
    trainingSet = []
    validSet = []
    if onTheFly:
        print("trying to init on the fly TrainSimulator thread")
        simulatorThread = advLR.Simulator(dataQueue, trainingSet, modelname, epochs, lr, batchSize,pastTS)
        print("trying to start on the fly TrainSimulator thread")
        simulatorThread.start()
        print("on the fly TrainSimulator thread started")
    else:
        allTrainData = glob('thesis/logData/adaptive/system*.csv')
        allValidData = glob('thesis/logData/valid/system*.csv')
        sorted(allTrainData, key=os.path.getmtime)
        first = True
        for each in allTrainData:
            if os.path.getsize(each) > 500000:
                loadedTSet = genfromtxt(each, delimiter=',', skip_header=1)
                print(each, loadedTSet.shape)
                if first == True:
                    trainingSet = loadedTSet
                    first = False
                else:
                    trainingSet = np.append(trainingSet, loadedTSet, axis=0)
        first = True
        for each in allValidData:
            if os.path.getsize(each) > 500000:
                loadedVSet = genfromtxt(each, delimiter=',', skip_header=1)
                print(each, loadedVSet.shape)
                if first == True:
                    validSet = loadedVSet
                    first = False
                else:
                    validSet = np.append(validSet, loadedVSet, axis=0)
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

def trainActor(actorQueueUser, actorQueueSimu, trainingSet):
    try:
        print("trying to init trainActor thread")
        actorTF = myActorCritic.ActorTrain(actorQueueUser, actorQueueSimu, trainingSet)
        print("trying to start trainActor thread")
        actorTF.start()
        print("trainActor thread started")
    except:
        print("unable to start trainActor thread")

if __name__ == '__main__':
    main()