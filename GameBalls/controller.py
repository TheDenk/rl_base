from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, Input, Bidirectional
from keras.models import model_from_json
from keras import optimizers

import keras
import numpy as np
import random as rnd


class Controller:

    def __init__(self):
        self.model_name = 'controller_model'
        self.model = create_model(32)
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

    def create_model(self, input_size):
        model = Sequential()
        model.add(Dense(128, input_dim=input_size, activation='sigmoid'))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dense(3, activation='linear'))
        model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])

        self.model = model
        return model

