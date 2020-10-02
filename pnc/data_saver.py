import os

import numpy as np
import pickle


class MetaSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton,
                                        cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DataSaver(metaclass=MetaSingleton):
    """
    Data Saver:
        create --> add topics --> advance
    """
    def __init__(self):
        self._history = dict()
        self._max_size = 1e5
        self._idx = 0
        if not os.path.exists('data'):
            os.makedirs('data')
        for f in os.listdir('data'):
            os.remove('data/' + f)

    def create(self, key, size):
        self._history[key] = np.zeros([self._max_size, size])

    def add(self, key, value):
        self._history[key][self._idx] = value

    def advance(self):
        if self._idx == (self._max_size - 1):
            self.save()
            self._idx = 0
        else:
            self._idx += 1

    def save(self):
        for (k, v) in self._history.items():
            np.save('data/' + k + '.npy', v, 'ab')
