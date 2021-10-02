'''
Author: Badri Adhikari, University of Missouri-St. Louis, 12-18-2019
File: Contains tensorflow models for the DEEPCON architecture
'''

import tensorflow as tf
from tensorflow.python.keras import layers

from tensorflow.python.keras.layers import Input, Convolution2D, Activation, add, Dropout, BatchNormalization
from tensorflow.python.keras.models import Model

# A basic fully convolutional network
def basic_fcn(L, num_blocks, width, expected_n_channels):
    input_shape = (L, L, expected_n_channels)
    img_input = layers.Input(shape = input_shape)
    x = img_input
    for i in range(num_blocks):
        x = layers.Conv2D(width, (3, 3), padding = 'same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    x = layers.Conv2D(1, (3, 3), padding = 'same', kernel_initializer = 'one')(x)
    x = layers.Activation('relu')(x)
    inputs = img_input
    model = tf.keras.models.Model(inputs, x, name = 'fcn')
    return model

# Architecture DEEPCON (original)
def deepcon_rdd(L, num_blocks, width, expected_n_channels):
    print('')
    print('Model params:')
    print('L', L)
    print('num_blocks', num_blocks)
    print('width', width)
    print('expected_n_channels', expected_n_channels)
    print('')
    dropout_value = 0.3
    my_input = Input(shape = (L, L, expected_n_channels))
    tower = BatchNormalization()(my_input)
    tower = Activation('relu')(tower)
    tower = Convolution2D(width, 1, padding = 'same')(tower)
    n_channels = width
    d_rate = 1
    for i in range(num_blocks):
        block = BatchNormalization()(tower)
        block = Activation('relu')(block)
        block = Convolution2D(n_channels, kernel_size = (3, 3), padding = 'same')(block)
        block = Dropout(dropout_value)(block)
        block = Activation('relu')(block)
        block = Convolution2D(n_channels, kernel_size = (3, 3), dilation_rate=(d_rate, d_rate), padding = 'same')(block)
        tower = add([block, tower])
        if d_rate == 1:
            d_rate = 2
        elif d_rate == 2:
            d_rate = 4
        else:
            d_rate = 1
    tower = BatchNormalization()(tower)
    tower = Activation('relu')(tower)
    tower = Convolution2D(1, 3, padding = 'same')(tower)
    tower = Activation('sigmoid')(tower)
    model = Model(my_input, tower)
    return model

# Architecture DEEPCON (distances)
def deepcon_rdd_distances(L, num_blocks, width, expected_n_channels):
    print('')
    print('Model params:')
    print('L', L)
    print('num_blocks', num_blocks)
    print('width', width)
    print('expected_n_channels', expected_n_channels)
    print('')
    dropout_value = 0.3
    my_input = Input(shape = (L, L, expected_n_channels))
    tower = BatchNormalization()(my_input)
    tower = Activation('relu')(tower)
    tower = Convolution2D(width, 1, padding = 'same')(tower)
    n_channels = width
    d_rate = 1
    for i in range(num_blocks):
        block = BatchNormalization()(tower)
        block = Activation('relu')(block)
        block = Convolution2D(n_channels, kernel_size = (3, 3), padding = 'same')(block)
        block = Dropout(dropout_value)(block)
        block = Activation('relu')(block)
        block = Convolution2D(n_channels, kernel_size = (3, 3), dilation_rate=(d_rate, d_rate), padding = 'same')(block)
        tower = add([block, tower])
        if d_rate == 1:
            d_rate = 2
        elif d_rate == 2:
            d_rate = 4
        else:
            d_rate = 1
    tower = BatchNormalization()(tower)
    tower = Activation('relu')(tower)
    tower = Convolution2D(1, 3, padding = 'same')(tower)
    tower = Activation('relu')(tower)
    model = Model(my_input, tower)
    return model

# Architecture DEEPCON (binned)
def deepcon_rdd_binned(L, num_blocks, width, bins, expected_n_channels):
    print('')
    print('Model params:')
    print('L', L)
    print('num_blocks', num_blocks)
    print('width', width)
    print('expected_n_channels', expected_n_channels)
    print('')
    dropout_value = 0.3
    my_input = Input(shape = (L, L, expected_n_channels))
    tower = BatchNormalization()(my_input)
    tower = Activation('relu')(tower)
    tower = Convolution2D(width, 1, padding = 'same')(tower)
    n_channels = width
    d_rate = 1
    for i in range(num_blocks):
        block = BatchNormalization()(tower)
        block = Activation('relu')(block)
        block = Convolution2D(n_channels, kernel_size = (3, 3), padding = 'same')(block)
        block = Dropout(dropout_value)(block)
        block = Activation('relu')(block)
        block = Convolution2D(n_channels, kernel_size = (3, 3), dilation_rate=(d_rate, d_rate), padding = 'same')(block)
        tower = add([block, tower])
        if d_rate == 1:
            d_rate = 2
        elif d_rate == 2:
            d_rate = 4
        else:
            d_rate = 1
    tower = BatchNormalization()(tower)
    tower = Activation('relu')(tower)
    tower = Convolution2D(bins, 3, padding = 'same')(tower)
    tower = Activation('softmax')(tower)
    model = Model(my_input, tower)
    return model
