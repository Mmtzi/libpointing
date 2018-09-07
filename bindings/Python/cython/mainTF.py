import _thread
import myFitsLawGame, SimulatorTest
from ml import advLR, recurrentSim
from queue import Queue
import time

def main():
    #dataQueue with sample data from fitsLawGame, every Thread has Acces to
    dataQueue = Queue()
    #trainSimThread = trainSimulator(dataQueue)
    #gameThread = collectData(dataQueue)
    testSimThread = testSimulator()

def collectData(dataQueue):
    try:
        print("trying to init Game thread")
        gameThread = myFitsLawGame.Game(dataQueue)
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

def testSimulator():
    try:
        print("trying to init SimTest thread")
        simTestThread = SimulatorTest.SimTest()
        print("trying to start SimTest thread")
        simTestThread.start()
        print("SimTest thread started")
    except:
        print("unable to start SimTest thread")

if __name__ == '__main__':
    main()