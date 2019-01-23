from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, MaxPool2D, Flatten, Activation, Embedding
from keras.layers import LSTM, Input, Bidirectional
from keras.models import model_from_json
from keras import optimizers

import keras
import numpy as np
import random as rnd


class Agent:

    def __init__(self):
        self.model = None
        self.memory = []
        self.batch_size = 32
        self.gamma = 0.9
        self.eps = 0  # rnd.random()
        self.eps_min = 0.0001
        self.eps_step = 0.995
        self.model_name = None
        pass

    def create_conv_model(self, w, h):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(w, h, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(3, activation='linear'))
        model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])
        return model

    def create_dence_model(self, size):
        model = Sequential()
        model.add(Dense(size, input_dim=size, activation='sigmoid'))
        model.add(Dense(128, activation='sigmoid'))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dense(3, activation='linear'))
        model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])
        return model

    def create_lstm_model(self, size):
        model = Sequential()
        model.add(LSTM(8, input_shape=(size, 1)))
        model.add(Dense(3, activation='linear'))
        model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])
        return model

    def do_action(self, state):
        if np.random.rand() <= self.eps:
            return rnd.randrange(0, 3)
        act = self.model.predict(state)[0]
        lst = list(act)
        return lst.index(max(lst))

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
        if len(self.memory) > 40000:
            del self.memory[0]
        pass

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = rnd.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            # target_f[0] *= 0.1
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.eps > self.eps_min:
            self.eps = self.eps * self.eps_step
        pass

    def load_model(self, model, weights):
        json = open(model, 'r')
        model_json = json.read()
        json.close()
        self.model = model_from_json(model_json)
        self.model.load_weights(weights)
        self.model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])

    def save_model(self):
        model_json = self.model.to_json()
        json = open(self.model_name + '.json', 'w')
        json.write(model_json)
        json.close()
        self.model.save_weights(self.model_name + '.h5')
    
