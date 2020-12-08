# Receive network outputs with one set of weights

import os.path
import glob
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, InputLayer
from PIL import Image
from matplotlib import cm


def euclidean_l2(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))


samples_per_archive = 3200 #size
p = 40. #maximum allowed transition

input_shape = (128, 128, 2)
kernel_size = 3
pool_size = 2
filters = 64
dropout = 0.5

# Compile the same network as used in training
model = Sequential()
model.add(InputLayer(input_shape))
model.add(Conv2D(filters=filters, \
                 kernel_size=kernel_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=filters, \
                 kernel_size=kernel_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters, \
                 kernel_size=kernel_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=filters, \
                 kernel_size=kernel_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters * 2, \
                 kernel_size=kernel_size, activation='relu', padding='same', ))
model.add(BatchNormalization())
model.add(Conv2D(filters=filters * 2, \
                 kernel_size=kernel_size, activation='relu', padding='same', ))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters * 2, \
                 kernel_size=kernel_size, activation='relu', padding='same', ))
model.add(BatchNormalization())
model.add(Conv2D(filters=filters * 2, \
                 kernel_size=kernel_size, activation='relu', padding='same', ))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(dropout))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(2))
model.summary()

sgd = optimizers.SGD(lr=0.005, momentum=0.9)

model.compile(loss=euclidean_l2, \
              optimizer=sgd, metrics=['mean_squared_error'])

# Load the weights which you want to use in the network
model.load_weights('<Path to weights>')

overall_sum = 0
predict_data_path = '<Path to dataset you want to test>'
archive = np.load(os.path.abspath(predict_data_path))
images = archive['images']
values = archive['values']

# Loop through the images and calculate errors for each transition
for idx in range(0, samples_per_archive):
    sample_image = images[idx]
    sample_shape = sample_image.shape
    sample_image = sample_image.reshape(1, 128, 128, 2)
    sample_values = values[idx]
    norm_sample_image = (sample_image - 127.5) / 127.5

    # The network outputs normalized values so they need to be multiplied by p
    norm_pred_values = model.predict(norm_sample_image)
    pred_values = norm_pred_values * p

    x_error = sample_values[0] - pred_values[0][0]
    y_error = sample_values[1] - pred_values[0][1]

    # Distance of how far away the transition is from the correct one
    euclidean_error = np.sqrt(np.power(x_error, 2) + np.power(y_error, 2))

    overall_sum += euclidean_error

mean_error = overall_sum / samples_per_archive

# In case you want to test the same dataset with different sets of weights, use
# model.load_weights('<Path to weights>')
# and loop through the images again