import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape

import numpy as np
import cv2


keras.backend.set_image_dim_ordering('th')


def crop_and_resize(image):
    cropped = image[300:650,500:,:]
    return cv2.resize(cropped, (448,448))


def normalize(image):
    normalized = 2.0*image/255.0 - 1
    return normalized


def preprocess(image):
    cropped = crop_and_resize(image)
    normalized = normalize(cropped)
    # The model works on (channel, height, width) ordering of dimensions
    transposed = np.transpose(normalized, (2,0,1))
    return transposed


def get_model():
    model = Sequential()

    # Layer 1
    model.add(Convolution2D(16, 3, 3 ,input_shape=(3 ,448 ,448) ,border_mode='same' ,subsample=(1 ,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    model.add(Convolution2D(32 ,3 ,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

    # Layer 3
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

    # Layer 4
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

    # Layer 5
    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

    # Layer 6
    model.add(Convolution2D(512, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

    # Layer 7
    model.add(Convolution2D(1024, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))

    # Layer 8
    model.add(Convolution2D(1024, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))

    # Layer 9
    model.add(Convolution2D(1024, 3, 3, border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Flatten())

    # Layer 10
    model.add(Dense(256))

    # Layer 11
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))

    # Layer 12
    model.add(Dense(1470))

    return model