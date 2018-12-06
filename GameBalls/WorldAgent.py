from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, MaxPool2D, Flatten, Activation, Embedding
from keras.layers import LSTM, Input, Bidirectional
from keras.models import model_from_json
from keras import optimizers

import keras
import numpy as np
import random as rnd

import vae
import controller

class Agent:

    def __init__(self):
        self.c_model = None
        self.vae_model = None
        self.rnn_model = None
        self.memory = []
        self.rnn_state_memory = []
        self.rnn_next_state_memory = []
        self.batch_size = 32
        self.gamma = 0.9
        self.eps = 0  # rnd.random()
        self.eps_min = 0.0001
        self.eps_step = 0.995

        self.hidden = np.zeros(self.rnn.hidden_units)
        self.cell_values = np.zeros(self.rnn.cell_values)
        self.action = 0
        self.c_actions = []
        self.c_rewards = []
        pass

    def do_action(self, state):
        if np.random.rand() <= self.eps:
            return rnd.randrange(0, 3)
        vae_encoded_obs = self.vae_model.encoder.predict(state)
        controller_obs = np.concatenate([vae_encoded_obs, self.hidden])
        action = self.c_model.predict(controller_obs)[0]

        return np.argmin(action)

    def update_rnn(self, next_state):
        vae_encoded_obs = self.vae_model.encoder.predict(next_state)
        input_to_rnn = [np.array([[np.concatenate([vae_encoded_obs, self.action])]]), np.array([self.hidden]),
                        np.array([self.cell_values])]
        h, c = self.rnn_model.forward.predict(input_to_rnn)
        self.hidden = h[0]
        self.cell_values = c[0]

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > 40000:
            del self.memory[0]
        pass

    def replay_controller(self):
        minibatch = zip(self.rnn_state_memory, self.c_actions, self.c_rewards, self.rnn_next_state_memory)
        for state, action, reward, next_state in minibatch:
            next_controller_obs = np.concatenate([next_state, self.hidden])
            target = reward + self.gamma * np.amax(self.c_model.predict(next_controller_obs)[0])
            controller_obs = np.concatenate([state, self.hidden])
            target_f = self.c_model.predict(controller_obs)
            # target_f[0] *= 0.1
            target_f[0][action] = target
            self.c_model.fit(state, target_f, epochs=1, verbose=0)
        if self.eps > self.eps_min:
            self.eps = self.eps * self.eps_step
        pass

    def replay_vae(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = rnd.sample(self.memory, self.batch_size)
        state = [state for state, action, reward, next_state in minibatch]
        state = np.array(state)

        next_state = [next_state for state, action, reward, next_state in minibatch]
        next_state = np.array(next_state)

        self.vae_model.train(state)

        actions = [action for state, action, reward, next_state in minibatch]
        actions = np.array(actions)
        self.rnn_state_memory = self.vae_model.generate_rnn_data(state, actions)
        self.rnn_next_state_memory = self.vae_model.generate_rnn_data(next_state, actions)
        self.c_actions = [action for state, action, reward, next_state in minibatch]
        self.c_rewards = [reward for state, action, reward, next_state in minibatch]

    def replay_rnn(self):
        self.rnn_model.train(self.rnn_state_memory[0], self.rnn_state_memory[1])
        pass

    def init_models(self):
        self.c_model = controller.Controller()
        self.vae_model = vae.VAE()
        self.rnn_model = rnn.RNN()
        pass

