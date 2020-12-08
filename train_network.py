# Training the network
# Most of the code is from Richard Guinto's implementation which can be found here:
# https://github.com/richard-guinto/homographynet

import os.path
import glob

import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, InputLayer
from keras.callbacks import ModelCheckpoint, ProgbarLogger
from keras import backend as K
from keras import optimizers
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# p = maximum allowed transition
def data_loader(path, batch_size=64, p=10.):
    while True:
        for npz in glob.glob(os.path.join(path, '*.npz')):
            archive = np.load(npz, allow_pickle=True)
            images = archive['images']
            values = archive['values']
            for i in range(0, len(values), batch_size):
                end_i = i + batch_size
                try:
                    batch_images = images[i:end_i]
                    batch_offsets = values[i:end_i]
                except IndexError:
                    continue
                # Normalize
                batch_images = (batch_images - 127.5) / 127.5
                batch_offsets = batch_offsets / p
                yield batch_images, batch_offsets


def euclidean_l2(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))


train_data_path = '<Path to folder containing the training dataset>'
samples_per_archive = 3200 # Here the amount of image pairs in the training dataset
num_archives = 1
num_samples = num_archives * samples_per_archive

batch_size = 64
total_iterations = 90000

steps_per_epoch = num_samples / batch_size
epochs = 60

input_shape = (128, 128, 2)
kernel_size = 3
pool_size = 2
filters = 64
dropout = 0.5

# Add layers to the Network
model = Sequential()
model.add(InputLayer(input_shape))
model.add(Conv2D(filters=filters,\
        kernel_size=kernel_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=filters,\
        kernel_size=kernel_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters,\
        kernel_size=kernel_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=filters,\
        kernel_size=kernel_size, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters*2,\
        kernel_size=kernel_size, activation='relu', padding='same',))
model.add(BatchNormalization())
model.add(Conv2D(filters=filters*2,\
        kernel_size=kernel_size, activation='relu', padding='same',))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size))
model.add(Conv2D(filters=filters*2,\
        kernel_size=kernel_size, activation='relu', padding='same',))
model.add(BatchNormalization())
model.add(Conv2D(filters=filters*2,\
        kernel_size=kernel_size, activation='relu', padding='same',))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(dropout))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(2))
model.summary()

sgd = optimizers.SGD(lr=0.005, momentum=0.9)

# Compile Network model
model.compile(loss=euclidean_l2,\
        optimizer=sgd, metrics=['mean_squared_error'])

# In case you want to continue training with same dataset, include this line with the latest network weights
# model.load_weights("<Path to network weights>")

# Save weights after each epoch
filepath = "<Path where to save epoch>" #Include epoch number in the file name by including {epoch:02d}
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1)
callback_list = [checkpoint]

# Start training
print('TRAINING...')
model.fit_generator(data_loader(train_data_path, batch_size),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs, callbacks=callback_list)

