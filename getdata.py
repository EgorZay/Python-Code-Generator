import os
import sys
import time
import json
import numpy as np

from logger import Logger

log = Logger('GetData')


class GetData(object):
    """
    Runs through the 'dataset' subfolder and searches for any .py script.
    Extracts every character and dumps it into a '.json' database
    called 'dataset.dt'. '.json' file is divided into three
    categories: ['chars', 'char_indices', 'indices_char']:
            chars: set of presented characters in 'dataset.dt';
            char_indices: dictionary of `chars` and according indices;
            indices_char: inverted `char_indices`. Used as a mapping variable.
    Proceeds then to transform and vectorize the data into ready-to-be-fit `X` and `y` inputs.
    """
    dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')
    dataset_file = os.path.join(dataset_dir, 'dataset.dt')

    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model')

    def __init__(self):
        """
        Default variable initialization method.
        """
        self.content = ''

        self.chars = []
        self.char_indices = {}
        self.indices_char = {}
        self.sentences = []
        self.next_chars = []
        # self.encoded = []

        # self.len_encoded = 0
        # self.num_batches = 0
        self.X = []
        self.y = []

    def get_dataset(self):
        """
        Creates a 'dataset' subfolder and parses for available .py scripts.
        Dumps their content into 'dataset.dt'.
        :return:
                chars: list
                    Sorted list of text characters.
                char_indices: dictionary
                    Character mapping.
                indices_char: dictionary
                    Inverted character mapping. Used to transform
                    the model output.
        """
        if not os.path.exists(self.dataset_dir):
            log.info("'dataset' subfolder not found.")
            os.mkdir(self.dataset_dir)
            log.info("'dataset' subfolder successfully created.")

        files = os.listdir(self.dataset_dir)

        if len(files) == 0:
            sys.exit('No scripts present.')

        names = np.array([
            file for file in files if file.find('.py') != -1
        ])
        print("'.py' scripts found: ", len(names))

        for name in names:
            with open(os.path.join(self.dataset_dir, name), 'r') as f:
                self.content += f.read() + '\n'
        log.info("'content' variable successfully built.")
        # vocab = set(content)
        # vocab_indices = {c: i for i, c in enumerate(vocab)}
        # indices_vocab = dict(enumerate(vocab))
        # encoded = np.array([vocab_indices[py] for py in content], dtype=np.int32)
        self.chars = sorted(list(set(self.content)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        # self.encoded = np.array([self.char_indices[char] for char in self.content], dtype=np.int32)
        # self.len_encoded = self.encoded.__len__()

        print('Total characters', len(self.chars))
        print('Total content length', len(self.content))
        # print('Total encoded length', len(self.encoded))
        time.sleep(0.5)

        if not os.path.isfile(self.dataset_file):
            log.info("'dataset.dt' database not found.")
            with open(self.dataset_file, 'w+') as d:
                # json_file = json.dumps({
                # 'encoded': [int(i) for i in encoded], 'vocab_indices': vocab_indices, 'indices_vocab': indices_vocab
                # })
                json_file = json.dumps({
                    'chars': [i for i in self.chars],
                    'char_indices': self.char_indices,
                    'indices_char': self.indices_char
                })
                d.write(json_file)
                d.close()
                log.info("'dataset.dt' database successfully built.")

        return self.chars, self.char_indices, self.indices_char

    def preprocess(self, maxlen=100, step=1):
        """

        :param maxlen:
        :param step:
        :return:
        """
        for i in range(0, len(self.content) - maxlen, step):
            self.sentences.append(self.content[i: i + maxlen])
            self.next_chars.append(self.content[i + maxlen])

        self.X = np.zeros((len(self.sentences), maxlen, len(self.chars)),
                          dtype=np.int32)
        self.y = np.zeros((len(self.sentences), len(self.chars)),
                          dtype=np.int32)

        for i, sentence in enumerate(self.sentences):
            for t, char in enumerate(sentence):
                self.X[i, t, self.char_indices[char]] = 1
            self.y[i, self.char_indices[self.next_chars[i]]] = 1

        print('=' * 50)
        print('< Quick data information >')
        print('X.shape', self.X.shape)
        print('y.shape', self.y.shape)
        print('X[0]', self.X[0])
        print('y[0]', self.y[0])

        return self.X, self.y

