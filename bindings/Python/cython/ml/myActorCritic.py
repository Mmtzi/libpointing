"""
solving pendulum using actor-critic model
"""
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Conv1D, BatchNormalization,Flatten, regularizers
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
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
    def __init__(self, queueUser, queueSimu, trainingSet, actorname):
        self.batchSize = 32
        self.epochs = 20
        self.actorname = actorname
        self.lr = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau = .125

        if trainingSet.size == 0:
            pass
        else:
            # 'dx', 'dy', 'button', 'rx', 'ry', 'time', 'distance',
            # 'directionX', 'directionY', 'targetX', 'targetY', 'targetSize', 'initMouseX', 'initMouseY', 'targetID'
            # env: dx,dy, button, time-1, distance-1, dirtx-1, dirty-1, targetx, targety, size
            self.outRxRy = trainingSet[:,[3,4]]
            self.shiftUp = trainingSet[:,[5,6,7,8]]
            self.shiftUp = np.roll(self.shiftUp, -1, axis =0)
            self.env = trainingSet[:,[0,1,2,5,6,7,8,9,10]]
            self.env[:,[3,4,5,6]] = self.shiftUp
            self.env = self.env[:-1,:]
            self.loadActorModel()
        super().__init__()


    def run(self):

        self.actorModel.fit_generator(generator=self.generator(),
                                      steps_per_epoch=int(self.env.shape[0]/self.batchSize),
                                      epochs=self.epochs,
                                      verbose=1,
                                      validation_data=self.validGenerator(),
                                      validation_steps=int(self.validInputNP.shape[0]/self.batchSize),
                                      callbacks=[self.tbCallBack, self.chk])
        try:
            self.actorModel.save('ml\\models\\actor\\'+str(self.actorname))
            print("saved model: "+str(str(self.actorname)))
        except:
            print("couldnt save model: "+str(self.actorname))

        K.clear_session()

        pass

    def loadActorModel(self):
        if os.path.exists('ml\\models\\actor\\' + str(self.actorname)):
            self.actorModel = load_model('ml\\models\\actor\\' + str(self.actorname))
            print("loaded model: " + str(self.actorname))
            print(K.get_value(self.actorModel.optimizer.lr))
            K.set_value(self.actorModel.optimizer.lr, self.lr)
        else:
            print("new model: " + str(self.actorname))
            stateInput = Input(shape=(self.env.shape[1]), dtype='float32', name='timeInput')
            norm = BatchNormalization()(stateInput)
            dense1 = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.0001))(norm)
            norm = BatchNormalization()(dense1)
            dense2 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(norm)
            outRxRy = Dense(2, activation='linear')(dense2)
            self.actorModel = Model([stateInput], [outRxRy])
            self.actorModel.compile(Adam(lr=self.lr), loss=['mse'])
        self.actorModel.summary()
        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #

    def generator(self):
        print("generateTrainDataSets")
        self.indexList = np.arange(0, self.env[0], 1)
        print(self.env.shape[0], self.outRxRy.shape[0], self.indexList.size)
        input = np.zeros((self.batchSize, self.env.shape[1]))
        outRxRy = np.zeros((self.batchSize, self.outRxRy.shape[1]))
        pickedIndex = np.zeros((self.batchSize, 1))
        print(input.shape, outRxRy.shape)
        while True:
            if self.indexList.shape[0] >= self.batchSize:
                yield self.createTrainBatch(input, outRxRy, pickedIndex)
            else:
                self.indexList = np.arange(0, self.env.shape[0], 1)

    def createTrainBatch(self, input, outdxdy, outbutton, pickedIndex):
        for i in range(0, self.batchSize):
            index = randint(0, self.indexList.shape[0] - 1)
            pick = self.indexList.item(index)
            pickedIndex[i] = index
            input[i] = self.inputNP[pick]
            outdxdy[i] = self.outDxDyNP[pick]
            outbutton[i] = self.outButtonNP[pick]
            # print(i, index, pick, input[i], outdxdy[i], outbutton[i])
        self.indexList = np.delete(self.indexList, pickedIndex)
        return [input], [outdxdy, outbutton]










































        #
        self.actor_model = self.create_actor_model()
        self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32,
                                                [None, self.env.action_space.shape[
                                                    0]])  # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights, -self.actor_critic_grad)  # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_action_input, \
        self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)  # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())


    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        #env: dx,dy, button, posx, posy, distx, disty, targetx, targety, size, time,
        state_input = Input(shape=self.env.observation_space.shape)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        #act: rx,ry
        output = Dense(self.env.action_space.shape[0], activation='relu')(h3)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        #env: dx,dy, button, posx, posy, distx, disty, targetx, targety, size, time
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)
        #act: ry, ry
        action_input = Input(shape=self.env.action_space.shape)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        #validation score
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action], reward, verbose=0)

    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.actor_model.predict(cur_state)


def main():
    sess = tf.Session()
    K.set_session(sess)
    env = (100,100)
    actor_critic = ActorCritic(env, sess)

    num_trials = 10000
    trial_len = 500

    cur_state = env.reset()
    action = env.action_space.sample()
    while True:
        cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
        action = actor_critic.act(cur_state)
        action = action.reshape((1, env.action_space.shape[0]))

        new_state, reward, done, _ = env.step(action)
        print(action)
        print(new_state, reward, done)
        new_state = new_state.reshape((1, env.observation_space.shape[0]))

        actor_critic.remember(cur_state, action, reward, new_state, done)
        actor_critic.train()

        cur_state = new_state