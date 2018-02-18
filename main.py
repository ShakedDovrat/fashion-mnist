import sys
import logging
import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard

from utils import mnist_reader


class Logger(object):
    def __init__(self, log_file_name):
        self.terminal = sys.stdout
        Logger._set_logger(log_file_name)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        logging.info(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

    @staticmethod
    def _set_logger(log_file_name):
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file_name)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


class Config(object):
    def __init__(self):
        self.image_size = (28, 28, 1)
        self.model = third_model_global_average_pooling
        self.model_name = self.model.__name__


def get_logger_filename():
    logger = logging.getLogger()
    handler = logger.handlers[0]
    return handler.baseFilename


def main():
    model_name = get_model_name()
    log_file_name = 'logs/{}.log'.format(model_name)
#     with Logger(log_file_name) as sys.stdout:
    sys.stdout = Logger(log_file_name)
    c = Config()
    model = build_model(c)
    compile_model(model)
    train_set, test_set = load_data(c.image_size)
    history = train(model, train_set)
    plot_training_history(history, model_name)
    test(model, test_set)


def get_model_name():
    return datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')


def set_logger(model_name):
    log_file_name = 'logs/{}.log'.format(model_name)



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

    return {'x': x, 'y': y}


def build_model(c):
    model = c.model(c.image_size)

    print('Using {}:'.format(c.model_name))
    model.summary()

    return model


def first_model(image_size):
    img_input = Input(image_size)

    x = Conv2D(16, 3, 3, border_mode='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(16, 3, 3, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Activation('relu')(x)

    x = Conv2D(32, 3, 3, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(32, 3, 3, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(100)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(10)(x)
    x = Activation('softmax')(x)

    return Model(img_input, x)


def second_model_BN_after_relu(image_size):
    img_input = Input(image_size)

    x = Conv2D(16, 3, 3, border_mode='same')(img_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(16, 3, 3, border_mode='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, 3, 3, border_mode='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(100)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(10)(x)
    x = Activation('softmax')(x)

    return Model(img_input, x)


def third_model_global_average_pooling(image_size):
    img_input = Input(image_size)

    x = Conv2D(16, 3, 3, border_mode='same')(img_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(16, 3, 3, border_mode='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, 3, 3, border_mode='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    #
    # x = Conv2D(64, 3, 3, border_mode='same')(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(100)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(10)(x)
    x = Activation('softmax')(x)

    return Model(img_input, x)


def shallow_model(image_size):
    img_input = Input(image_size)

    x = Conv2D(16, 3, 3, border_mode='valid')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(10)(x)
    x = Activation('softmax')(x)

    return Model(img_input, x)


def compile_model(model):
    loss = 'sparse_categorical_crossentropy' # 'categorical_crossentropy'
    optimizer = Adam(lr=1e-4)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])


def get_callbacks():
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=0.1,
                                  patience=5,
                                  verbose=1,
                                  mode='auto')
    checkpoint_writer = ModelCheckpoint('weights.h5',
                                        monitor='val_acc',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='max')
    tensor_board = TensorBoard(log_dir='logs',
                               histogram_freq=1,
                               write_graph=True,
                               write_images=False)
    early_stop = EarlyStopping(monitor='val_acc',
                               min_delta=0,
                               patience=20,
                               verbose=1,
                               mode='auto')

    return [reduce_lr, checkpoint_writer, tensor_board, early_stop]


def train(model, dataset):
    callbacks = get_callbacks()
    history = model.fit(dataset['x'],
                        dataset['y'],
                        batch_size=32,
                        nb_epoch=200,
                        verbose=1,
                        callbacks=callbacks,
                        validation_split=0.2)
    return history


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


def f(msg):
    print(msg)


def test(model, dataset):
    metrics_list = model.evaluate(dataset['x'], dataset['y'], verbose=1)
    print('Test results:')
    [f('{} = {}'.format(name, value)) for name, value in zip(model.metrics_names, metrics_list)]


if __name__ == '__main__':
    main()
