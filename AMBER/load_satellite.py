import os
import numpy as np
import pickle

from keras.utils.np_utils import to_categorical   
def load_satellite_data(path, train):
    train_file = os.path.join(path, 'satellite_train.npy')
    test_file = os.path.join(path, 'satellite_test.npy')

    all_train_data, all_train_labels = np.load(train_file, allow_pickle=True)[()]['data'], np.load(train_file,allow_pickle=True)[()]['label']
    test_data, test_labels = np.load(test_file, allow_pickle=True)[()]['data'], np.load(test_file, allow_pickle=True)[()]['label']

    #rerange labels to 0-23
    all_train_labels = all_train_labels - 1
    test_labels = test_labels - 1
    all_train_labels = to_categorical(all_train_labels, num_classes=24)
    test_labels = to_categorical(test_labels, num_classes=24)

    #normalize data
    all_train_data = (all_train_data - all_train_data.mean(axis=1, keepdims=True))/all_train_data.std(axis=1, keepdims=True)
    test_data = (test_data - test_data.mean(axis=1, keepdims=True))/test_data.std(axis=1, keepdims=True)

    #add dimension
    all_train_data = np.expand_dims(all_train_data, 2)
    test_data = np.expand_dims(test_data, 2)


    if train:
        len_val = len(test_data)
        train_data = all_train_data[:-len_val]
        train_labels = all_train_labels[:-len_val]
        val_data, val_labels = all_train_data[-len_val:], all_train_labels[-len_val:]


        return train_data, train_labels, val_data, val_labels


    return all_train_data, all_train_labels, test_data, test_labels
