import numpy as np
from math import sqrt, acos, log2
from queue import Queue
import time
from random import randint
class Environment():
    def __init__(self, queueUser, queueSimu):
        self.qUser = queueUser
        self.qSimu = queueSimu
        self.action_space = np.zeros(shape=2, )
        self.observation_space = np.zeros(shape=8, )
        self.training = False
        self.updateSampleList()

    def updateSampleList(self):
        while self.qUser.qsize() < 20:
            print(self.qUser.qsize())
            time.sleep(0.5)
        self.generateActorSamples()
        while self.training:
            time.sleep(0.01)
        self.setSampleAsObs()

    def setSampleAsObs(self):
        sampleLine = randint(0, self.myObservationSamples.shape[0])
        self.observation_space = self.myObservationSamples[sampleLine-1, :]
        self.action_space = self.outRxRy[sampleLine-1, :]
        self.training = True

    def generateActorSamples(self):
        # 'dx', 'dy', 'button', 'rx', 'ry', 'time', 'distance', 'targetID'
        # 'directionX', 'directionY', 'targetX', 'targetY', 'targetSize', 'initMouseX', 'initMouseY', 'size', 'score'
        # state: dx,dy, button, time-1, distance-1, dirtx-1, dirty-1, targetx, targety, size
        # act: rx, ry
        rawSampleArray = np.array(list(self.qUser.queue))
        print(rawSampleArray[0])
        self.outRxRy = rawSampleArray[1:, [3, 4]]
        print(self.outRxRy[0])
        self.shiftDown = rawSampleArray[:, [6, 8, 9]]
        print(self.shiftDown[0])
        self.shiftDown = np.roll(self.shiftDown, 1, axis=0)
        print(self.shiftDown[0])
        self.myObservationSamples = rawSampleArray[:, [0, 1, 6, 8, 9, 10, 11, 14]]
        print(self.myObservationSamples[0])
        self.myObservationSamples[:, [ 3, 4, 5]] = self.shiftDown
        print(self.myObservationSamples[0])
        self.myObservationSamples = self.myObservationSamples[1:, :]
        print(self.myObservationSamples[0])

    def step(self, action):
        self.training = False
        newState =  self.calcNewState(action)
        reward = self.calcReward(action)
        return newState, -reward, False, {}

    def reset(self):
        return self.calcNewState(np.zeros(shape=2,))

    def calcNewState(self, action):
        #print(self.observation_space[5], action)
        if action.shape == (1,2):
            action = action.reshape(2,)
            #print(action.shape)
        self.observation_space[3] = self.observation_space[3] + action[0]
        self.observation_space[4] = self.observation_space[4] + action[0]
        self.observation_space[2] = sqrt(pow(self.observation_space[2], 2)+ pow(self.observation_space[2],2))
        return self.observation_space

    def calcReward(self,action):
        if action.shape == (1,2):
            action = action.reshape(2,)
            #print(action.shape)
        old_beeline = self.observation_space[5] - self.observation_space[3], self.observation_space[6] - self.observation_space[4]
        new_beeline = self.observation_space[5]+action[0] - self.observation_space[3], self.observation_space[6]+action[1]- self.observation_space[4]
        print(old_beeline, new_beeline)
        sk_old_new = old_beeline[0] * new_beeline[0] + old_beeline[1] * new_beeline[1]
        sk_old_mvt = old_beeline[0] * action[0] + old_beeline[1] * action[1]
        length_old_beeline = sqrt(pow(old_beeline[0], 2) + pow(old_beeline[1], 2))
        length_new_beeline = sqrt(pow(new_beeline[0], 2) + pow(new_beeline[1], 2))
        length_movement = sqrt(pow(action[0], 2) + pow(action[1], 2))
        if length_movement > 0 and length_old_beeline > 0:
            angle_atStart = np.abs(np.rad2deg(acos(min(abs(sk_old_mvt) / (length_old_beeline * length_movement), 1))))
        else:
            angle_atStart = 15
        if length_old_beeline > 0 and length_new_beeline > 0:
            angle_atTarget = np.abs(
                np.rad2deg(acos(min(abs(sk_old_new) / (length_old_beeline * length_new_beeline), 1))))
        else:
            angle_atTarget = 15
        return angle_atTarget+angle_atStart
