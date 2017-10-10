import os
PROJECT_HOME = '/home/carnd/CarND-Behavioral-Cloning'
DRIVING_LOG = os.path.join(PROJECT_HOME, 'data', 'driving_log.csv')
IMG_DIR = os.path.join(PROJECT_HOME, 'data', 'IMG')

import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import json
import cv2

from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution1D, Convolution2D

def yield_train_tuples():
    with open(DRIVING_LOG, 'r') as fh:
        # skip header
        fh.readline()
        for line in fh:
            centerImgPath, leftImgPath, rightImgPath, steering, _throttle, _brake, _speed = line.split(',')
            steering = float(steering)
            adjust = 0.2
            for imgPath, steer in ((centerImgPath, steering), (leftImgPath, steering + adjust), (rightImgPath, steering - adjust)):
                image = Image.open(os.path.join(PROJECT_HOME, 'data', imgPath.strip()))
                image_array = np.asarray(image)
                transformed_image_array = normalize_and_shape(image_array)
                # horizontal flip
                yield transformed_image_array, steer
                yield np.fliplr(transformed_image_array), -steer
            
def normalize_and_shape(img):
    img = cv2.resize(img[50:-30,], (160, 40), interpolation = cv2.INTER_CUBIC)
    #img = img.astype('float32')
    #ing = img / 255 - 0.5
    return img

def main():
    from sklearn.model_selection import train_test_split
    train_tup = list(yield_train_tuples())
    features, labels = zip(*train_tup)
    X_train, X_val, Y_train, Y_val = train_test_split(np.array(features), np.array(labels), random_state=0, test_size=0.33)

    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_train = X_train / 255 - 0.5
    X_val = X_val / 255 -0.5

    # define and compile model
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, input_shape=(40, 160, 3)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(80, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))


    model.summary()
    LEARNING_RATE=0.001
    sgd = keras.optimizers.SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)
    # Compile and train the model
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['accuracy'])

    # generator used for fit_generator
    def data_walker(X_array, Y_array, step=100):
        while 1:
            for i in range(0, int(len(X_array) / step)):
                yield(X_array[i*step:(i+1)*step], Y_array[i*step:(i+1)*step])

    NBR_TRAIN = len(X_train)
    N_EPOCH = 10
    SAMPLES_PER_EPOCH = NBR_TRAIN# int(NBR_TRAIN / N_EPOCH)

    history = model.fit_generator(
        data_walker(X_train, Y_train, 100),
        samples_per_epoch=SAMPLES_PER_EPOCH,
        nb_epoch=N_EPOCH,
        verbose=1,
        validation_data=(X_val, Y_val),
    )

    # save the model and weights
    with open('model.2.json', 'w') as fh:
        fh.write(model.to_json())
    model.save_weights('model.2.h5')



if __name__ == '__main__':
    main()


