from __future__ import print_function

import os

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM

from getdata import GetData
from logger import Logger

log = Logger('Model')


class Model(object):
    """
    Builds simple LSTM model using 'keras' library with 'tensorflow` backend.
    The structure of the LSTM includes 128 hidden units, dense layer and a softmax activation layer.
    """
    gd = GetData()
    gd.get_dataset()

    def __init__(self):
        """
        Default variable initialization method.
        """
        self.char_len = len(self.gd.chars)

        self.maxlen = self.gd.preprocess.__defaults__[0]

        self.model = Sequential()

    def lstm_(self, units=512):
        """

        :param units:
                positive int, optional
                    Dimensionality of the output space
        :return:
                object
                    Returns an object of class `keras.models.Sequential`.
        """
        if not os.path.exists(self.gd.model_dir):
            log.info("LSTM-model subfolder not found.")
            os.mkdir(self.gd.model_dir)
            log.info("'model' subfolder created.")

        if os.path.exists(self.gd.model_dir):
            if os.path.exists(os.path.join(self.gd.model_dir, 'model.h5')):
                log.info("LSTM-model found.")
                self.model = load_model(os.path.join(self.gd.model_dir, 'model.h5'))
                self.model.summary()

                return self.model

            else:
                log.info("Previous state not found. Initializing LSTM-model structure. Stand by ...")
                self.model.add(LSTM(units=units, input_shape=(self.maxlen, self.char_len),
                                    dropout=0.2, return_sequences=False))  # set return_sequences to True to stack LSTMs
                #self.model.add(Dropout(0.5))
                #self.model.add(LSTM(units=units, dropout=0.2,
                #                    return_sequences=False))
                self.model.add(Dense(units=self.char_len))
                self.model.add(Activation(activation='softmax'))

                log.info("LSTM-model successfully initialized.")
                self.model.summary()

        return self.model

