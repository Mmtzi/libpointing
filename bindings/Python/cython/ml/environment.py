import numpy as np
from math import sqrt, acos, log2
from queue import Queue
from threading import Thread
import time
class Environment(Thread):
    def __init__(self, queueUser, queueSimu, pastTS):
        self.qUser = queueUser
        self.qSimu = queueSimu
        self.sampleQueue = Queue()
        self.pastTimeSteps = pastTS
        self.action_space = np.zeros(shape=2, )
        self.observation_space = np.zeros(shape=24, )
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau = .125
        super().__init__()
    def run(self):
        while self.qUser.qsize() < 500:
            print(self.qUser.qsize())
            time.sleep(0.5)
        self.createStrokeData()

    def createStrokeData(self):
        # 'dx', 'dy', 'button', 'rx', 'ry', 'time', 'distance', 'targetID'
        # 'directionX', 'directionY', 'targetX', 'targetY', 'initMouseX', 'initMouseY', 'size', 'score'
        # env: dx,dy, button, time-1, distance-1, dirtx-1, dirty-1, targetx, targety, size
        # act: rx, ry
        # rew: score
        #(math.log2(math.sqrt(pow((self.targetPosition[0]-self.initCursorPos[0]), 2) + pow(self.targetPosition[1]-self.initCursorPos[1], 2))/self.pointSize*2), self.newTimestamp)
        data = np.array(list(self.qUser.queue))
        i=0
        j=0
        while i+1 < data.shape[0]:
            while data[i,7] == data[i+1,7]:
                i+=1
                if i+1 >= data.shape[0]:
                    break
            else:
                strokeData = data[j:i+1,:]
                j = i + 1
                i = j
                strokeData = self.calcScoreForStroke(strokeData)
                self.generateActorSamples(strokeData)

    def calcScoreForStroke(self, strokeData):
        print(strokeData.shape)
        indexOfDifficulty = log2(sqrt(pow(strokeData[0,10]-strokeData[0,12], 2) + pow(strokeData[0,11]-strokeData[0,13],2))/strokeData[0,14])
        throughput = indexOfDifficulty/strokeData[strokeData.shape[0]-1,5]
        #addOnwScore
        for i in range(0, strokeData.shape[0]-1):
            strokeData[i,15] = throughput
        return strokeData

    def generateActorSamples(self, strokeData):
        for i in range(self.pastTimeSteps, strokeData.shape[0]):
            batchsample = strokeData[i - self.pastTimeSteps: i + 1, [0,1,3,4]]
            #print(batchsample)
            batchsample = batchsample.flatten()
            print(batchsample)
            self.sampleQueue.put(batchsample)

        # 'dx', 'dy', 'button', 'rx', 'ry', 'time', 'distance', 'targetID'
        # 'directionX', 'directionY', 'targetX', 'targetY', 'targetSize', 'initMouseX', 'initMouseY', 'size', 'score'
        # state: dx,dy, button, time-1, distance-1, dirtx-1, dirty-1, targetx, targety, size
        # act: rx, ry
        # rew: score


    def step(self, action):
        return self.calcNewState(action), self.calcReward(action), False, {}

    def reset(self):
        return self.calcNewState(np.zeros(shape=2,))

    def calcNewState(self, action):
        self.pastTimeSteps = 5
        self.action_space = np.zeros(shape=2, )
        self.observation_space = np.zeros(shape=24, )
        self.old_observation_space = self.observation_space
        #print(self.observation_space[5], action)
        if action.shape == (1,2):
            action = action.reshape(2,)
            #print(action.shape)
        index = int(self.pastTimeSteps*4+2)
        self.observation_space[index] = self.observation_space[index]+action[0]
        self.observation_space[index+1] = self.observation_space[index+1]+action[1]
        return self.old_observation_space

    def calcReward(self,action):
        if action.shape == (1,2):
            action = action.reshape(2,)
            #print(action.shape)
        old_beeline = self.old_observation_space[7] - self.old_observation_space[5], self.old_observation_space[8] - self.old_observation_space[6]
        new_beeline = self.observation_space[7] - self.observation_space[5], self.observation_space[8] - self.observation_space[6]
        movement = action[0], action [1]
        sk_old_new = old_beeline[0] * new_beeline[0] + old_beeline[1] * new_beeline[1]
        sk_old_mvt = old_beeline[0] * movement[0] + old_beeline[1] * movement[1]
        length_old_beeline = sqrt(pow(old_beeline[0], 2) + pow(old_beeline[1], 2))
        length_new_beeline = sqrt(pow(new_beeline[0], 2) + pow(new_beeline[1], 2))
        length_movement = sqrt(pow(movement[0], 2) + pow(movement[1], 2))
        if length_movement > 0 and length_old_beeline > 0:
            angle_atStart = np.abs(np.rad2deg(acos(min(abs(sk_old_mvt) / (length_old_beeline * length_movement), 1))))
        else:
            angle_atStart = 1
        if length_old_beeline > 0 and length_new_beeline > 0:
            angle_atTarget = np.abs(
                np.rad2deg(acos(min(abs(sk_old_new) / (length_old_beeline * length_new_beeline), 1))))
        else:
            angle_atTarget = 1
        return angle_atTarget+angle_atStart

    #
    # distFkt = pow(log2(distance / 100 + 1.01) + 1, 2)
    #
    # movement = length_old_beeline - length_new_beeline
    # if (movement) <= 0:
    #     movementToTargetPenalty = abs(movement)
    # else:
    #     movementToTargetPenalty = 1 / (movement)
    # movementToTargetPenalty = (movementToTargetPenalty * targetSize)
    # if button and distance <= targetSize:
    #     hit = 10
    # else:
    #     hit = 0
    # print("angleAtStart: " + str(round(angle_atStart, 2)) + "  angleAtTarget: " + str(
    #     round(angle_atTarget, 2)) + "  distance: " + str(round(distance, 2))
    #       + "  distFkt: " + str(round(distFkt, 2)) + "  rx: " + str(
    #     round(actCursorPos[0] - oldCursorPos[0], 2)) + "  ry: " + str(round(actCursorPos[1] - oldCursorPos[1], 2))
    #       + "  movement2targetp: " + str(round(movementToTargetPenalty, 2)) + "  size: " + str(
    #     targetSize) + "  time: " + str(time))
    #
    # score = round(
    #     log2(100 / (angle_atStart + angle_atTarget * 10 + 1)) + (10 / (movementToTargetPenalty + 1)) * distFkt, 2) + hit
