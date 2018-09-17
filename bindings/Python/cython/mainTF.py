import _thread
import myFitsLawGame, SimulatorTest
from ml import advLR, recurrentSim, actorTF
from queue import Queue
#import itertools,collections
import time

def main():
    #dataQueue with sample data from fitsLawGame, every Thread has Acces to
    simuQueue = Queue()
    actorQueueUser = Queue()
    actorQueueSimu = Queue()
    trainSimThread = trainSimulator(simuQueue)
    gameThread = collectData(simuQueue, actorQueueUser)
    #testSimThread = testSimulator(actorQueueSimu)

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



def trainSimulator(dataQueue):
    try:
        print("trying to init Simulator thread")
        simulatorThread = advLR.Simulator(dataQueue)
        print("trying to start Simulator thread")
        simulatorThread.start()
        print("Simulator thread started")
    except:
        print("unable to start Simulator thread")

def testSimulator(actorQueue):
    try:
        print("trying to init SimTest thread")
        simTestThread = SimulatorTest.SimTest(actorQueue)
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