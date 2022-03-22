from typing import Callable, Optional, Union

import torch
import pytorch_lightning as pl
import numpy as np
import pickle
import os
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split

@DATAMODULE_REGISTRY
class ECGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        channels_last: bool = True,
        data_dir: Optional[str] = ".cache",
        num_workers: int = 3,
        batch_size: int = 64,
        pin_memory: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self._image_shape = [1, 1, 1_000] # TODO
        self.num_classes = 4
        self.batch_size = batch_size

        if channels_last:
            self._image_shape = self._image_shape[1], self._image_shape[2], self._image_shape[0]

    def prepare_data(self):
        read_data_physionet_4_with_val('../datasets/ecg')

    def setup(self, stage):
        self.ecg_train, self.ecg_valid, self.ecg_test = \
            read_data_physionet_4_with_val('../datasets/ecg')

    def train_dataloader(self):
        return DataLoader(
            self.ecg_train, 
            batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(
            self.ecg_valid, 
            batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(
            self.ecg_test, 
            batch_size=self.batch_size, shuffle=False, num_workers=8)

    @property
    def image_shape(self):
        return self._image_shape

    def default_transforms(self) -> Callable:
        return ecg_transform(channels_last=self.hparams.channels_last)

def ecg_transform(channels_last: bool = True):
    transform_list = []

    def channels_to_last(img: torch.Tensor):
        return img.permute(1, 2, 0).contiguous()

    transform_list.append(transforms.ToTensor())

    if channels_last:
        transform_list.append(channels_to_last)

    return transforms.Compose(transform_list)


class ECGDataset(Dataset):
    def __init__(self, data, label, pid=None):
        self.data = data
        self.label = label
        self.pid = pid

    def __getitem__(self, index):
        return (torch.tensor(self.data[index][:, :, np.newaxis], 
                dtype=torch.float), 
            torch.tensor(self.label[index], 
                dtype=torch.long))

    def __len__(self):
        return len(self.data)

def read_data_physionet_4_with_val(path, window_size=1000, stride=500):
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
