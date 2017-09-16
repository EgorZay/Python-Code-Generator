import os
import time
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import save_model, load_model
from keras.optimizers import RMSprop

from logger import Logger
from getdata import GetData
from model import Model

log = Logger('Train')


class Train(object):
    """
    Trains the input model on vectorized `X` and `y` inputs.
    Uses RMSProp as an optimizer with 5e-4 learning rate value.
    Saves checkpoint weights after each epoch.
    By default, trains on batches of size 256.
    """
    gd = GetData()
    gd.get_dataset()

    model = Model()

    def __init__(self):
        """
        Default variable initialization method.
        """
        self.X, self.y = self.gd.preprocess()

        self.model = self.model.lstm_()
        self.learning_rate = 5e-4
        self.optimizer = RMSprop(lr=self.learning_rate)

        self.model_dir = self.gd.model_dir
        self.model_ = os.path.join(self.model_dir, 'model.h5')
        self.ckpt_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'weights.hdf5')

        self.content = self.gd.content
        self.maxlen = self.gd.preprocess.__defaults__[0]

        self.checkpointer = ModelCheckpoint(filepath=self.ckpt_dir,
                                            verbose=1, save_best_only=False)
        self.earlystopping = EarlyStopping(monitor='loss', min_delta=0.01, patience=3, verbose=1, mode='auto')

    def fit(self):
        """

        :return:
                object
                    Returns an object of class `keras.models.Sequential`.
        """
        if os.path.exists(self.model_):
            log.info('Fully-trained model found.')
            self.model = load_model(self.model_)

            return self.model

        elif os.path.exists(self.ckpt_dir):
            log.info('LSTM-model checkpoint found.')
            self.model = load_model(self.ckpt_dir)

            self.model.fit(self.X, self.y,
                           batch_size=256, epochs=20,
                           callbacks=[self.checkpointer, self.earlystopping])

            return self.model

        else:
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=self.optimizer)
            log.info("Commencing model fitting. Stand by ...")
            time.sleep(0.5)

            # for i in range(1, 30):
                # print('Iteration: ', i)
            self.model.fit(self.X, self.y,
                           batch_size=256, epochs=20,
                           callbacks=[self.checkpointer,
                                      self.earlystopping])
            log.info("LSTM-model successfully fitted.")
            save_model(self.model, filepath=self.model_)
            log.info("LSTM-model dumped at 'model'.")

            return self.model

    @staticmethod
    def sample(preds, temp=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temp
        preds_exp = np.exp(preds)
        preds = preds_exp / np.sum(preds_exp)
        probas = np.random.multinomial(1, preds, 1)

        return np.argmax(probas)

if __name__ == '__main__':
    tr = Train()
    tr.fit()
