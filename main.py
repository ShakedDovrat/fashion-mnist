from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D, Dense
from keras import backend as K
from keras.models import Model
import keras.utils as ku
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    model = build_model()

    (x_train2, y_train2), (x_test2, y_test2) = fashion_mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # x_train2 = np.expand_dims(x_train, 3)
    # y_train2 = ku.to_categorical(y_train)
    # x_test2 = np.expand_dims(x_test, 3)
    # y_test2 = ku.to_categorical(y_test)

    history = model.fit(x=x_train2, y=y_train2, batch_size=256, epochs=100, validation_split=0.2)

    scores = model.evaluate(x=x_test2, y=y_test2)
    print("test results:")
    for i in range(len(scores)):
        print("{}: {}".format(model.metrics_names[i], scores[i]))

    plot_history(history)


def build_model():
    input_img = Input(shape=(28, 28))

    layer_1 = Conv2D(10, (3, 3), padding='same', activation='relu')(input_img)
    layer_2 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_1)

    layer_3 = Conv2D(10, (3, 3), padding='same', activation='relu')(layer_2)
    layer_4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_3)

    layer_5 = Conv2D(10, (3, 3), padding='same', activation='relu')(layer_4)
    layer_6 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_5)

    layer_7 = GlobalMaxPooling2D()(layer_6)
    layer_8 = Dense(10, activation=K.softmax)(layer_7)

    model = Model(inputs=input_img, outputs=layer_8)
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    return model

def plot_history(history):
    # list all data in history
    print(history.history.keys())

    # summarize history for accuracy
    plt.figure(1)
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

    plt.show()


if __name__ == '__main__':
    main()
