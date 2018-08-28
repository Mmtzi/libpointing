import _thread
import myFitsLawGame
from ml import advLR, recurrentSim
from queue import Queue

def main():
    dataQueue = Queue()
    collectData(dataQueue)
    trainSimulator(dataQueue)



def collectData(dataQueue):
    try:
        print("trying to init Game thread")
        gameThread = myFitsLawGame.Game(dataQueue)
        print("trying to start Game thread")
        gameThread.start()
        print("Game thread started")

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


if __name__ == '__main__':
    main()