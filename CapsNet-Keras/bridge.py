import numpy as np
from keras.utils import to_categorical

import utils.mnist_reader


def load_fashion_mnist():
    return load_single_set('train'), load_single_set('t10k')


def load_single_set(kind):
    x, y = utils.mnist_reader.load_mnist('../data/fashion', kind=kind)
    x = np.reshape(x, (x.shape[0], 28, 28, 1))
    y = np.reshape(y, (y.shape[0], 1))
    y = to_categorical(y)
    return x, y
