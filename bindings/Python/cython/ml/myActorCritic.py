"""
solving pendulum using actor-critic model
"""
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Conv1D, BatchNormalization,Flatten, regularizers
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam, SGD
import keras.backend as K
import tensorflow as tf
import os
import random
from collections import deque
from threading import Thread
from random import randint

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic(Thread):
    def __init__(self, queueUser, queueSimu, trainingSet, actorname, criticname):
        self.batchSize = 32
        self.epochs = 20
        self.actorname = actorname
        self.criticname = criticname
        self.lr = 0.001
        self.gamma = .95
        self.tau = .125

        if trainingSet.size == 0:
            pass
        else:
            # 'dx', 'dy', 'button', 'rx', 'ry', 'time', 'distance', 'targetID'
            # 'directionX', 'directionY', 'targetX', 'targetY', 'targetSize', 'initMouseX', 'initMouseY', 'size', 'score'
            # env: dx,dy, button, time-1, distance-1, dirtx-1, dirty-1, targetx, targety, size
            # act: rx, ry
            # rew: score
            print(trainingSet.shape)
            self.outRxRy = trainingSet[1:,[3,4]]
            self.shiftDown = trainingSet[:,[5,6,7,8]]
            self.shiftDown = np.roll(self.shiftDown, 1, axis =0)
            self.reward = trainingSet[1:,15]
            self.env = trainingSet[:,[0,1,2,5,6,7,8,9,10,]]
            self.env[:,[3,4,5,6]] = self.shiftDown
            self.env = self.env[1:,:]
            self.sess = tf.Session()
            K.set_session(self.sess)

            #actor
            self.actorStateInput, self.actorModel = self.loadActorModel()
            _, self.targetActorModel = self.loadActorModel()
            print(self.outRxRy.shape[1])
            self.actorCriticGrad = tf.placeholder(tf.float32, [None, self.outRxRy.shape[1]])
            actorModelWeights = self.actorModel.trainable_weights
            self.actorGrads = tf.gradients(self.actorModel.output, actorModelWeights, -self.actorCriticGrad)
            grads = zip(self.actorGrads, actorModelWeights)
            self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)

            #critic
            self.criticStateInput, self.criticActionInput, self.criticModel = self.loadCriticModel()
            _,_, self.targetCriticModel = self.loadCriticModel()
            self.criticGrads = tf.gradients(self.criticModel.output, self.criticActionInput)
            self.sess.run(tf.initialize_all_variables())

        super().__init__()


    def train(self):
        batch_size = 32
        print("training..")
        print(self.env.shape)
        if self.env.shape[0] < batch_size:
            return
        self.indexList = np.arange(0, self.env.shape[0]-self.batchSize, 1)
        np.random.shuffle(self.indexList)
        self.IterPickIndex = 0
        for i in range(0, self.batchSize):
            pickedIndex = self.indexList[self.IterPickIndex]
            print(self.env.shape, self.outRxRy.shape, self.reward.shape)
            myEnvBatch = self.env[pickedIndex: pickedIndex+self.batchSize,:]
            myActBatch = self.outRxRy[pickedIndex: pickedIndex+self.batchSize,:]
            myRewardBatch = self.reward[pickedIndex: pickedIndex+self.batchSize,]
            self.criticModel.fit([myEnvBatch,myActBatch], myRewardBatch, verbose=2)
            predicted_action = self.actorModel.predict(self.env[pickedIndex])
            grads = self.sess.run(self.criticGrads, feed_dict={
                self.criticStateInput: self.env[pickedIndex],
                self.criticActionInput: predicted_action
            })[0]
            print(self.env[pickedIndex])
            print(predicted_action)
            self.sess.run(self.optimize, feed_dict={
                self.actorStateInput: self.env[pickedIndex],
                self.actorCriticGrad: grads
            })
            self.IterPickIndex += 1

    def _update_actor_target(self):
        actor_model_weights = self.actorModel.get_weights()
        actor_target_weights = self.targetCriticModel.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.targetCriticModel.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.criticModel.get_weights()
        critic_target_weights = self.targetCriticModel.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.targetCriticModel.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    def act(self, cur_state):
        return self.actorModel.predict(cur_state)



    def run(self):
        i=0
        while True:
            #currentState = self.env[[i],:]
            #currentAction = self.outRxRy[[i],:]
            self.train()
            #self.update_target()
            #action = self.act(currentState)
            #reward = self.criticModel.predict(currentState, currentAction)
            #self.train()
            #i+=1

        K.clear_session()

    def loadActorModel(self):
        print("new model: " + str(self.actorname))
        print(None, self.env.shape[1])
        stateInput = Input(batch_shape=(None, self.env.shape[1]), dtype=tf.float32)

        dense1 = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(stateInput)
        dense2 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(dense1)
        outRxRy = Dense(2, activation='linear')(dense2)
        actorModel = Model([stateInput], [outRxRy])
        actorModel.compile(Adam(lr=self.lr), loss=['mse'])
        actorModel.summary()

        return stateInput, actorModel


    def loadCriticModel(self):
        print("new model: " + str(self.criticname))
        print(self.env.shape)
        stateInput = Input(batch_shape=(None, self.env.shape[1]), dtype=tf.float32)

        denseState1 = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(stateInput)
        denseState2 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(denseState1)
        actionInput = Input(batch_shape=(None, self.outRxRy.shape[1]), dtype=tf.float32)
        denseAct1 = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(actionInput)
        merged = Add()([denseState2, denseAct1])
        denseMerged1 = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(merged)
        outReward = Dense(1, activation='relu')(denseMerged1)
        criticModel = Model([stateInput, actionInput], [outReward])
        criticModel.compile(Adam(lr=self.lr), loss=['mse'])
        criticModel.summary()

        return stateInput, actionInput, criticModel