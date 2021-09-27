# -*- coding: utf-8 -*-
import h5py
import numpy as np
import tensorflow as tf
import scipy.io as sio

from tqdm import tqdm

def serialize_example(x, y):
    # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    example = {
        'x': tf.train.Feature(int64_list=tf.train.Int64List(value=x.flatten())),
        'y': tf.train.Feature(int64_list=tf.train.Int64List(value=y.flatten()))}

    # Create a Features message using tf.train.Example.
    example = tf.train.Features(feature=example)
    example = tf.train.Example(features=example)
    serialized_example = example.SerializeToString()
    return serialized_example

def traindata_to_tfrecord():
    filename = '../deepsea_filtered.npz'
    with np.load(filename) as file:
        x = file['x_train'] # shape = (71753, 1000, 4)
        y = file['y_train'] # shape = (71753, 36)

    for file_num in range(1):
        with tf.io.TFRecordWriter('./data/traindata-%.2d.tfrecord' % file_num) as writer:
            for i in tqdm(range(file_num*71753, (file_num+1)*71753), desc="Processing Train Data {}".format(file_num), ascii=True):
                example_proto = serialize_example(x[i], y[i])
                writer.write(example_proto)

def testdata_to_tfrecord():
    filename = '../deepsea_filtered.npz'
    data = np.load(filename)
    x = data['x_test'] # shape = (149400, 1000, 4)
    y = data['y_test'] # shape = (149400, 36)

    with tf.io.TFRecordWriter('./data/testdata.tfrecord') as writer:
        for i in tqdm(range(len(y)), desc="Processing Test Data", ascii=True):
            example_proto = serialize_example(x[i], y[i])
            writer.write(example_proto)

if __name__ == '__main__':
    # Write the train data and test data to .tfrecord file.
    traindata_to_tfrecord()
    #testdata_to_tfrecord()
