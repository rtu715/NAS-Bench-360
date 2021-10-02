# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.io as sio

def parse_function(example_proto):
    dics = {
        'x': tf.io.FixedLenFeature([1000, 4], tf.int64),
        'y': tf.io.FixedLenFeature([36], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, dics)
    x = tf.reshape(parsed_example['x'], [1000, 4])
    y = tf.reshape(parsed_example['y'], [36])
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    return (x, y)

def get_train_data(batch_size):
    filenames = ['./data/traindata-00.tfrecord']
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=100000, num_parallel_reads=4)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset 

def get_valid_data():
    data = np.load('../deepsea_filtered.npz')
    x = data['x_val']  # shape = (2490, 1000, 4)
    y = data['y_val']  # shape = (2490, 36)
    return (x, y)

def get_test_data():
    filename = '../deepsea_filtered.npz'
    data = np.load(filename)
    x = data['x_test'].astype(float)  # shape = (149400, 1000, 4)
    y = data['y_test']  # shape = (149400, 36)
    return (x, y)
