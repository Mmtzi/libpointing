import _thread
import myFitsLawGame, SimulatorTest
from ml import advLR, recurrentSim, actorTF
from queue import Queue
from numpy import genfromtxt
from glob import glob
from random import shuffle
import os
#import itertools,collections
import time

def main():
    modelname="sim_tconvAdv_fitg_20dx_20dist_sizeo.h5"
    #dataQueue with sample data from fitsLawGame, every Thread has Acces to
    simuQueue = Queue()
    actorQueueUser = Queue()
    actorQueueSimu = Queue()
    trainAll = True
    i=0
    while trainAll and i <10:
        allCSVData = glob('thesis/logData/adaptive/system*.csv')
        shuffle(allCSVData)
        for each in allCSVData:
            if os.path.getsize(each) >400000:
                trainingSet = genfromtxt(each, delimiter=',', skip_header=1)
                print(each, trainingSet.shape)
                myTrainThread = trainSimulator(simuQueue, trainingSet, modelname)
                time.sleep(10)
        time.sleep(30)
        i+=1
    #trainSimThread = trainSimulator(simuQueue, trainingSet, modelname)
    #gameThread = collectData(simuQueue, actorQueueUser)
    testSimThread = testSimulator(actorQueueSimu, modelname)

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



def trainSimulator(dataQueue, trainingSet, modelname):
    try:
        print("trying to init TrainSimulator thread")
        simulatorThread = advLR.Simulator(dataQueue, trainingSet, modelname)
        print("trying to start TrainSimulator thread")
        simulatorThread.start()
        print("TrainSimulator thread started")
    except:
        print("unable to start TrainSimulator thread")
        return

    if trainingSet.size !=0:
        simulatorThread.join()

def testSimulator(actorQueue, modelname):
    try:
        print("trying to init SimTest thread")
        simTestThread = SimulatorTest.SimTest(actorQueue, modelname)
        print("trying to start SimTest thread")
        simTestThread.start()
        print("SimTest thread started")
    except:
        print("unable to start SimTest thread")

def trainActor(actorQueueUser, actorQueueSimu):
    try:
        print("trying to init trainActor thread")
        actorTF = actorTF.ActorTrain(actorQueueUser, actorQueueSimu)
        print("trying to start trainActor thread")
        actorTF.start()
        print("trainActor thread started")
    except:
        print("unable to start trainActor thread")

if __name__ == '__main__':
    main()