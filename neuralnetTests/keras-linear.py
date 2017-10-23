# begin
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from tensorflow import set_random_seed
set_random_seed(1337)

from sys import exit
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Input, Reshape, Dense, Activation, Embedding
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.datasets import imdb
from keras.models import model_from_json

print('Loading data...')

X_train = np.array([[ 1, 0, 1 ]])
y_train = np.array([[ 1 ]])

X_test = X_train
y_test = y_train

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')

model = Sequential()
model.add(Dense(3, input_shape=(3,), activation='linear', bias=False))
model.add(Dense(1, activation='linear', bias=False))

print('Compile...')
sgd = SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd)

print('Evaluate 1...')
score = model.evaluate(X_test, y_test, batch_size=1, verbose=2)
print(model.metrics_names[0], ':', score)
prediction = model.predict(X_test, batch_size=1, verbose=2)
print('prediction:', prediction)


print('Pre-Train Weights...')
for layer in model.layers:
    weights = layer.get_weights()
    print(weights)

print('Train...')
model.fit(X_train, y_train, nb_epoch=1, batch_size=1, verbose=2)

print('Post-Train Weights...')
for layer in model.layers:
    weights = layer.get_weights()
    print(weights)

print('Evaluate 2...')
score = model.evaluate(X_test, y_test, batch_size=1, verbose=2)
print(model.metrics_names[0], ':', score)
prediction = model.predict(X_test, batch_size=1, verbose=2)
print('prediction:', prediction)

