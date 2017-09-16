import os
import sys
import time
import random
import numpy as np

from logger import Logger
from getdata import GetData
from model import Model
from train import Train

from keras.models import load_model

log = Logger('Test')


class Test(object):
    """
    Loads  the model and proceeds to predict on a random input sequence of length `self.maxlen`.
    """
    gd = GetData()
    gd.get_dataset()

    tr = Train()

    def __init__(self):
        """
        Default variable initialization method.
        """
        self.chars = self.gd.chars
        self.char_indices = self.gd.char_indices
        self.indices_char = self.gd.indices_char
        self.content = self.gd.content

        self.model_dir = self.gd.model_dir
        self.model_ = os.path.join(self.model_dir, 'model.h5')
        self.model_ckpt = os.path.join(self.model_dir, '/tmp/weights.hdf5')

        self.maxlen = self.gd.preprocess.__defaults__[0]
        self.start_index = self.start_index = random.randint(0,
                                                             len(self.content) - self.maxlen - 1)

        self.X, self.y = self.gd.preprocess()

        self.checkpointer = self.tr.checkpointer

    def load_model(self):
        """
        Loads the model from the `self.model_dir`; or the last checkpoint from `self.model_ckpt`.
        :return:
                object
                    Returns an object of class `keras.models.Sequential`.
        """
        if os.path.exists(self.model_):
            model = load_model(self.model_)
            log.info("LSTM-model successfully loaded.")

            return model

        elif os.path.exists(self.model_ckpt):
            model = load_model(self.model_ckpt)
            # log.info("Continuing training from the checkpoint.")
            # time.sleep(0.5)
            # model.fit(self.X, self.y,
            #           batch_size=128, epochs=1,
            #           callbacks=[self.checkpointer])
            log.info("LSTM-model from the last checkpoint successfully loaded.")

            return model

        else:
            log.warning("Neither LSTM-model nor checkpoint weights found.")
            log.warning('Consider to fit the model first.')
            sys.exit('Abort.')

    def predict(self, model, script_len=50):
        """
        Predicts on a given input.
        :param model:
                object, keras.models.Sequential
                    Input LSTM-model to perform extrapolation.
        :param script_len:
                positive int, optional
                    The length of a given code to produce.
        :return:
                code: list
                    List of produced texts. The number of texts is the length of `diversity`
        """
        model_ = model
        log.info('LSTM-model successfully loaded.')

        code = []
        for diversity in [0.2, 0.8, 1.2]:  # leave it be for this instance
            print('\n\n')
            print('Generating script with given diversity: ', diversity)

            generated = ''
            sentence = self.content[self.start_index: self.start_index + self.maxlen]
            generated += sentence

            print('Generating script from starting point of length `self.maxlen`: ', sentence)
            print('\n\n')

            sys.stdout.write(generated)

            for i in range(script_len):
                x = np.zeros((1, self.maxlen, len(self.chars)))
                for t, char in enumerate(sentence):
                    x[0, t, self.char_indices[char]] = 1.  # do not comprehend

                    preds = model_.predict(x)[0]
                    next_index = self.tr.sample(preds, diversity)
                    next_char = self.indices_char[next_index]

                    generated += next_char
                    sentence = sentence[1:] + next_char

                    sys.stdout.write(next_char)
                    sys.stdout.flush()
            code.append(sentence)

        return code

if __name__ == '__main__':
    t = Test()
    model = t.load_model()
    py = t.predict(model, 50)
    print(py[0])
