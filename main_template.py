import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import datetime

import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.datasets import fashion_mnist

from models import *


class Config(object):
    def __init__(self):
        self.image_size = (28, 28, 1)
        self.model = shallow_model
        self.model_name = self.model.__name__
        self.lr = 1e-4
        self.batch_size = 32
        self.epochs = 5


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


def get_callbacks(model_name):
    checkpoint_writer = ModelCheckpoint('{}weights.h5'.format(model_name),
                                        monitor='val_acc',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='max')

    return [checkpoint_writer]


def train(model, c, dataset, model_name):
    callbacks = get_callbacks(model_name)
    history = model.fit(dataset['x'],
                        dataset['y'],
                        batch_size=c.batch_size,
                        epochs=c.epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_split=0.2)
    return history


def load_data():
    trainset, testset = fashion_mnist.load_data()
    return dataset_to_dict(trainset), dataset_to_dict(testset)


def dataset_to_dict(dataset):
    x, y = dataset
    return {'x': x, 'y': y}


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
    train_set, test_set = load_data()
    history = train(model, c, train_set, model_name)
    plot_training_history(history, model_name)


if __name__ == '__main__':
    main()
