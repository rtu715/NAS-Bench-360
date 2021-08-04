import os
import numpy as np
import torch
import pandas as pd
import scipy.io
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter

import torch.utils.data as data_utils
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, data, label, pid=None):
        self.data = data
        self.label = label
        self.pid = pid

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)


def load_data(task, path, train=False):
    if task == 'ECG':
        return load_ECG_data(path, train)

    elif task == 'satellite':
        return load_satellite_data(path, train)

    else:
        raise NotImplementedError

def load_ECG_data(path, train):
    return read_data_physionet_4_with_val(path) if train else read_data_physionet_4(path)

def load_satellite_data(path, train):
    train_file = os.path.join(path, 'satellite_train.npy')
    test_file = os.path.join(path, 'satellite_test.npy')

    all_train_data, all_train_labels = np.load(train_file, allow_pickle=True)[()]['data'], np.load(train_file,allow_pickle=True)[()]['label']
    test_data, test_labels = np.load(test_file, allow_pickle=True)[()]['data'], np.load(test_file, allow_pickle=True)[()]['label']

    #rerange labels to 0-23
    all_train_labels = all_train_labels - 1
    test_labels = test_labels - 1

    #normalize data
    all_train_data = (all_train_data - all_train_data.mean(axis=1, keepdims=True))/all_train_data.std(axis=1, keepdims=True)
    test_data = (test_data - test_data.mean(axis=1, keepdims=True))/test_data.std(axis=1, keepdims=True)

    #add dimension
    all_train_data = np.expand_dims(all_train_data, 1)
    test_data = np.expand_dims(test_data, 1)

    #convert to tensor/longtensor
    all_train_tensors, all_train_labeltensor = torch.from_numpy(all_train_data).type(torch.FloatTensor), \
                                               torch.from_numpy(all_train_labels).type(torch.LongTensor)

    test_tensors, test_labeltensor = torch.from_numpy(test_data).type(torch.FloatTensor), torch.from_numpy(test_labels).type(torch.LongTensor)
    testset = data_utils.TensorDataset(test_tensors, test_labeltensor)

    if train:
        len_val = len(test_data)
        train_tensors, train_labeltensor = all_train_tensors[:-len_val], all_train_labeltensor[:-len_val]
        val_tensors, val_labeltensor = all_train_tensors[-len_val:], all_train_labeltensor[-len_val:]

        trainset = data_utils.TensorDataset(train_tensors, train_labeltensor)
        valset = data_utils.TensorDataset(val_tensors, val_labeltensor)

        return trainset, valset, testset

    trainset = data_utils.TensorDataset(all_train_tensors, all_train_labeltensor)

    return trainset, None, testset

def read_data_physionet_4(path, window_size=3000, stride=500):

    # read pkl
    with open(os.path.join(path,'challenge2017.pkl'), 'rb') as fin:
        res = pickle.load(fin)
    ## scale data
    all_data = res['data']
    for i in range(len(all_data)):
        tmp_data = all_data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        all_data[i] = (tmp_data - tmp_mean) / tmp_std
    ## encode label
    all_label = []
    for i in res['label']:
        if i == 'N':
            all_label.append(0)
        elif i == 'A':
            all_label.append(1)
        elif i == 'O':
            all_label.append(2)
        elif i == '~':
            all_label.append(3)
    all_label = np.array(all_label)

    # split train test
    X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_label, test_size=0.1, random_state=0)

    # slide and cut
    print('before: ')
    print(Counter(Y_train), Counter(Y_test))
    X_train, Y_train = slide_and_cut(X_train, Y_train, window_size=window_size, stride=stride)
    X_test, Y_test, pid_test = slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride,
                                             output_pid=True)
    print('after: ')
    print(Counter(Y_train), Counter(Y_test))

    # shuffle train
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)

    trainset = ECGDataset(X_train, Y_train)
    testset = ECGDataset(X_test, Y_test, pid_test)

    return trainset, None, testset#, pid_test

def read_data_physionet_4_with_val(path, window_size=3000, stride=500):

    # read pkl
    with open(os.path.join(path,'challenge2017.pkl'), 'rb') as fin:
        res = pickle.load(fin)
    ## scale data
    all_data = res['data']
    for i in range(len(all_data)):
        tmp_data = all_data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        all_data[i] = (tmp_data - tmp_mean) / tmp_std
    ## encode label
    all_label = []
    for i in res['label']:
        if i == 'N':
            all_label.append(0)
        elif i == 'A':
            all_label.append(1)
        elif i == 'O':
            all_label.append(2)
        elif i == '~':
            all_label.append(3)
    all_label = np.array(all_label)

    # split train val test
    X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_label, test_size=0.2, random_state=0)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=0)

    # slide and cut
    print('before: ')
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))
    X_train, Y_train = slide_and_cut(X_train, Y_train, window_size=window_size, stride=stride)
    X_val, Y_val, pid_val = slide_and_cut(X_val, Y_val, window_size=window_size, stride=stride, output_pid=True)
    X_test, Y_test, pid_test = slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride,
                                             output_pid=True)
    print('after: ')
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))

    # shuffle train
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    X_test = np.expand_dims(X_test, 1)

    trainset = ECGDataset(X_train, Y_train)
    valset = ECGDataset(X_val, Y_val, pid_val)
    testset = ECGDataset(X_test, Y_test, pid_test)

    return trainset, valset, testset#, pid_val, pid_test

def slide_and_cut(X, Y, window_size, stride, output_pid=False, datatype=4):
    out_X = []
    out_Y = []
    out_pid = []
    n_sample = X.shape[0]
    mode = 0
    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]
        if tmp_Y == 0:
            i_stride = stride
        elif tmp_Y == 1:
            if datatype == 4:
                i_stride = stride//6
            elif datatype == 2:
                i_stride = stride//10
            elif datatype == 2.1:
                i_stride = stride//7
        elif tmp_Y == 2:
            i_stride = stride//2
        elif tmp_Y == 3:
            i_stride = stride//20
        for j in range(0, len(tmp_ts)-window_size, i_stride):
            out_X.append(tmp_ts[j:j+window_size])
            out_Y.append(tmp_Y)
            out_pid.append(i)
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)
