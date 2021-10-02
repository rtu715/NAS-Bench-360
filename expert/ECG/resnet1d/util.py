import numpy as np
import pandas as pd
import scipy.io
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

def preprocess_physionet():
    """
    download the raw data from https://physionet.org/content/challenge-2017/1.0.0/, 
    and put it in ../data/challenge2017/

    The preprocessed dataset challenge2017.pkl can also be found at https://drive.google.com/drive/folders/1AuPxvGoyUbKcVaFmeyt3xsqj6ucWZezf
    """
    
    # read label
    label_df = pd.read_csv('../data/challenge2017/REFERENCE-v3.csv', header=None)
    label = label_df.iloc[:,1].values
    print(Counter(label))

    # read data
    all_data = []
    filenames = pd.read_csv('../data/challenge2017/training2017/RECORDS', header=None)
    filenames = filenames.iloc[:,0].values
    print(filenames)
    for filename in tqdm(filenames):
        mat = scipy.io.loadmat('../data/challenge2017/training2017/{0}.mat'.format(filename))
        mat = np.array(mat['val'])[0]
        all_data.append(mat)
    all_data = np.array(all_data)

    res = {'data':all_data, 'label':label}
    with open('../data/challenge2017/challenge2017.pkl', 'wb') as fout:
        pickle.dump(res, fout)

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


def read_data_physionet_2_clean_federated(m_clients, test_ratio=0.2, window_size=3000, stride=500):
    """
    - only N A, no O P
    - federated dataset, evenly cut the entire dataset into m_clients pieces
    """

    # read pkl
    with open('../data/challenge2017/challenge2017.pkl', 'rb') as fin:
        res = pickle.load(fin)
    ## scale data
    all_data = res['data']
    for i in range(len(all_data)):
        tmp_data = all_data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        all_data[i] = (tmp_data - tmp_mean) / tmp_std
    all_data_raw = res['data']
    all_data = []
    ## encode label
    all_label = []
    for i in range(len(res['label'])):
        if res['label'][i] == 'A':
            all_label.append(1)
            all_data.append(res['data'][i])
        elif res['label'][i] == 'N':
            all_label.append(0)
            all_data.append(res['data'][i])
    all_label = np.array(all_label)
    all_data = np.array(all_data)

    # split into m_clients
    shuffle_pid = np.random.permutation(len(all_label))
    m_clients_pid = np.array_split(shuffle_pid, m_clients)
    all_label_list = [all_label[i] for i in m_clients_pid]
    all_data_list = [all_data[i] for i in m_clients_pid]

    out_data = []
    for i in range(m_clients):
        print('clinet {}'.format(i))
        tmp_label = all_label_list[i]
        tmp_data = all_data_list[i]

        # split train test
        X_train, X_test, Y_train, Y_test = train_test_split(tmp_data, tmp_label, test_size=test_ratio, random_state=0)
        
        # slide and cut
        print('before: ')
        print(Counter(Y_train), Counter(Y_test))
        X_train, Y_train = slide_and_cut(X_train, Y_train, window_size=window_size, stride=stride, datatype=2.1)
        X_test, Y_test, pid_test = slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride, datatype=2.1, output_pid=True)
        print('after: ')
        print(Counter(Y_train), Counter(Y_test))
        
        # shuffle train
        shuffle_pid = np.random.permutation(Y_train.shape[0])
        X_train = X_train[shuffle_pid]
        Y_train = Y_train[shuffle_pid]

        X_train = np.expand_dims(X_train, 1)
        X_test = np.expand_dims(X_test, 1)

        out_data.append([X_train, X_test, Y_train, Y_test, pid_test])

    return out_data


def read_data_physionet_4(window_size=1000, stride=500):

    # read pkl
    with open('challenge2017.pkl', 'rb') as fin:
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
    X_test, Y_test, pid_test = slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride, output_pid=True)
    print('after: ')
    print(Counter(Y_train), Counter(Y_test))
    
    # shuffle train
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)

    return X_train, X_test, Y_train, Y_test, pid_test

def read_data_physionet_4_with_val(window_size=1000, stride=500):

    # read pkl
    with open('challenge2017.pkl', 'rb') as fin:
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
    X_test, Y_test, pid_test = slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride, output_pid=True)
    print('after: ')
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))
    
    # shuffle train
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    X_test = np.expand_dims(X_test, 1)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, pid_val, pid_test


def read_data_generated(n_samples, n_length, n_channel, n_classes, verbose=False):
    """
    Generated data
    
    This generated data contains one noise channel class, plus unlimited number of sine channel classes which are different on frequency. 
    
    """    
    all_X = []
    all_Y = []
    
    # noise channel class
    X_noise = np.random.rand(n_samples, n_channel, n_length)
    Y_noise = np.array([0]*n_samples)
    all_X.append(X_noise)
    all_Y.append(Y_noise)
    
    # sine channel classe
    x = np.arange(n_length)
    for i_class in range(n_classes-1):
        scale = 2**i_class
        offset_list = 2*np.pi*np.random.rand(n_samples)
        X_sin = []
        for i_sample in range(n_samples):
            tmp_x = []
            for i_channel in range(n_channel):
                tmp_x.append(np.sin(x/scale+2*np.pi*np.random.rand()))
            X_sin.append(tmp_x)
        X_sin = np.array(X_sin)
        Y_sin = np.array([i_class+1]*n_samples)
        all_X.append(X_sin)
        all_Y.append(Y_sin)

    # combine and shuffle
    all_X = np.concatenate(all_X)
    all_Y = np.concatenate(all_Y)
    shuffle_idx = np.random.permutation(all_Y.shape[0])
    all_X = all_X[shuffle_idx]
    all_Y = all_Y[shuffle_idx]
    
    # random pick some and plot
    if verbose:
        for _ in np.random.permutation(all_Y.shape[0])[:10]:
            fig = plt.figure()
            plt.plot(all_X[_,0,:])
            plt.title('Label: {0}'.format(all_Y[_]))
    
    return all_X, all_Y


if __name__ == "__main__":
    read_data_physionet_2_clean_federated(m_clients=4)
