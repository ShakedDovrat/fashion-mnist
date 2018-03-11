import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from utils import mnist_reader
from models import *


class Config(object):
    def __init__(self):
        self.image_size = (28, 28, 1)
        self.model = ConvTwo_model  # shallow_model
        self.model_name = self.model.__name__
        self.lr = 1e-4
        self.batch_size = 64
        self.epochs = 20


def get_model_name():
    return datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')


def build_model(c):
    model = c.model(c.image_size)

    print('Using {}:'.format(c.model_name))
    model.summary()

    return model


def compile_model(model, c):
    loss = 'sparse_categorical_crossentropy' # 'categorical_crossentropy'
    optimizer = Adam(lr=c.lr)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])


def get_callbacks():
    checkpoint_writer = ModelCheckpoint('weights.h5',
                                        monitor='val_acc',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='max')

    return [checkpoint_writer]


def train(model, c, dataset):
    callbacks = get_callbacks()
    history = model.fit(dataset['x'],
                        dataset['y'],
                        batch_size=c.batch_size,
                        epochs=c.epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_split=0.2)
    return history


def load_data(image_size):
    return load_single_dataset('train', image_size), \
           load_single_dataset('t10k' , image_size)


def load_single_dataset(kind, image_size):
    x, y = mnist_reader.load_mnist('data/fashion', kind=kind)

    x_resize_shape = (-1,) + image_size
    x = np.reshape(x, x_resize_shape)
    y = np.reshape(y, (-1, 1))

    x = x.astype(np.float32)# / 255
    y = y.astype(np.float32)# / 255

    # ---------------

    arXY = np.c_[x.reshape(len(x), -1), y.reshape(len(y), -1)]
    np.random.shuffle(arXY)
    x2 = arXY[:, :x.size // len(x)].reshape(x.shape)
    y2 = arXY[:, x.size // len(x):].reshape(y.shape)


    return {'x': x2, 'y': y2}


def plot_training_history(history, model_name):
    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.savefig('logs/{}.png'.format(model_name))

    plt.show()


def test(model, dataset):
    metrics_list = model.evaluate(dataset['x'], dataset['y'], verbose=1)
    print('Test results:')
    for name, value in zip(model.metrics_names, metrics_list):
        print('{} = {}'.format(name, value))


def main():
    model_name = get_model_name()
    c = Config()
    model = build_model(c)
    compile_model(model, c)
    train_set, test_set = load_data(c.image_size)
    history = train(model, c, train_set)
    plot_training_history(history, model_name)


if __name__ == '__main__':
    main()
