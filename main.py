import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras.datasets import fashion_mnist
from keras.layers import Dense, Flatten
from keras.models import Sequential
import keras

# ToDo
# pLOT lOSS & ACCURACY
# validation
# Functions
# Add initialization
# Tune Learn rate 

# 60,000*28*28
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#ToDO  Build network


model = Sequential()

# Stacking layers is as easy as .add():

# input_dim=(28,28)
model.add(Flatten(input_shape=(28,28)))

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=10, activation='softmax'))


model.summary()


model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)