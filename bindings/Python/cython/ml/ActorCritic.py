
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
from ml import environment2

import tensorflow as tf

import random
from collections import deque
from threading import Thread


# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic(Thread):
    def __init__(self, queueUser, queueSimu):
        self.sess = tf.Session()
        K.set_session(self.sess)
        np.set_printoptions(precision=0,suppress=True)
        self.env = environment2.Environment(queueUser, queueSimu)

        num_trials = 10000
        trial_len = 500

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .5
        self.tau = .125

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #

        self.memory = deque(maxlen=2000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()
        self.actor_critic_grad = tf.placeholder(tf.float32,
                                                [None, self.env.action_space.shape[
                                                    0]])  # where we will feed de/dC (from critic)
        print("actor critic grads: placeholder shape: "+str(self.env.action_space.shape[0]))
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
        self.sess.run(tf.global_variables_initializer())
        self.graph = tf.get_default_graph()
        super().__init__()
    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        print("actor_stateInput shape:" + str(self.env.observation_space.shape))
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.env.action_space.shape[0], activation='relu')(h3)
        print("actor_output(action) shape:" + str(self.env.action_space.shape[0]))

        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        print("critic_stateInput shape:"+str(self.env.observation_space.shape))
        print(self.env.observation_space)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=self.env.action_space.shape)
        print("critic_actionInput shape:" + str(self.env.action_space.shape))
        action_h1 = Dense(48, name="criticactioninput")(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu', name="mergeddense")(merged)
        output = Dense(1, activation='relu', name="outcritic")(merged_h1)
        print("critic_output(evaluation) shape:" + str(output.shape))
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
        #print("training actor...")
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            #print("actor", sample)
            predicted_action = self.actor_model.predict(cur_state)
            print(cur_state, predicted_action)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        #print("training critic...")
        for sample in samples:
            #print("criticsample",sample)
            cur_state, action, reward, new_state, done = sample
            with self.graph.as_default():
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                #print(cur_state, new_state, target_action, future_reward)
                reward += self.gamma * future_reward
            #print("reward= ")
            #print(reward)
            with self.graph.as_default():
                self.critic_model.fit([cur_state, action], [reward], verbose=0)

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

            #self.env.action_space = np.array([random.randint(-5,5), random.randint(-5,5)])
            #print("action_space_sample:", self.env.action_space)
            return self.env.action_space
        #print("act" + str(cur_state.shape))
        with self.graph.as_default():
            return self.actor_model.predict(cur_state)

    def run(self):
        cur_state = self.env.reset()
        print("env reset:" + str(cur_state))
        action = self.env.action_space
        while True:
            cur_state = cur_state.reshape((1, self.env.observation_space.shape[0]))
            action = self.act(cur_state)
            #print(action.shape)
            action = action.reshape((1, self.env.action_space.shape[0]))
            #print(action.shape)
            new_state, reward, done, _ = self.env.step(action)
            #np.array([0.1, -0.1, 0.1]), np.array([-2]), False, {}
            print("step in acion:"+ str(action) +"return: newstate, reward, done"+str(new_state)+str( reward)+str( done))
            #print(str(action.shape))
            new_state = new_state.reshape((1, self.env.observation_space.shape[0]))

            self.remember(cur_state, action, reward, new_state, done)
            self.train()

            cur_state = new_state
            self.env.setSampleAsObs()