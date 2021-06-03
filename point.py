import os
import numpy as np
import torch
import shutil
from torch.autograd import Variable
import math
import gzip
import pickle
from torch import nn

import torch.utils.data as data_utils

import torchvision
from torchvision import transforms


def load_data(task, path, train=True, permute=False):
    '''use train=True to have validation split, otherwise set to false'''


    if task == 'scifar100':
        return load_scifar100_data(path, 0.2, train)

    elif task == 'ninapro':
        return load_ninapro_data(path, train)

    elif task == 'cifar100':
        trainset, valset = load_cifar100_train_data(path, permute, 0.2, train)
        testset = load_cifar100_test_data(path, permute)
        return trainset, valset, testset

    else:
        raise NotImplementedError



'''
Standard and Permuted CIFAR-100 related
'''
def bitreversal_permutation(n):
    log_n = int(math.log2(n))
    assert n == 1 << log_n, 'n must be a power of 2'
    perm = np.arange(n).reshape(n, 1)
    for i in range(log_n):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    perm = perm.squeeze(0)
    return torch.tensor(perm)

class RowColPermute(nn.Module):

    def __init__(self):
        super().__init__()
        #self.rowperm = torch.randperm(row) if type(row) == int else row
        #self.colperm = torch.randperm(col) if type(col) == int else col
        self.rowperm = bitreversal_permutation(32)
        self.colperm = bitreversal_permutation(32)
    
    def forward(self, tensor):
        return tensor[:, self.rowperm][:, :, self.colperm]



def load_cifar100_train_data(path, permute=False, val_split=0.2, train=True):
    #We could include cutout in transforms
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD = [0.2675, 0.2565, 0.2761]
    normalize = transforms.Normalize(CIFAR_MEAN,
                                     CIFAR_STD)

    if permute:
        permute_op = RowColPermute()
        transform = transforms.Compose([transforms.ToTensor(), permute_op, normalize])

    else:
        transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             normalize]
        )

    all_trainset = torchvision.datasets.CIFAR100(
        root=path, train=True, download=True, transform=transform
    )

    if val_split==0.0 or not train:
        return all_trainset, None

    n_train = int((1-val_split) * len(all_trainset))
    train_ind = torch.arange(n_train)
    val_ind = torch.arange(n_train, len(all_trainset))
    trainset = data_utils.Subset(all_trainset, train_ind)
    valset = data_utils.Subset(all_trainset, val_ind)

    return trainset, valset

def load_cifar100_test_data(path, permute=False):
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD = [0.2675, 0.2565, 0.2761]

    normalize = transforms.Normalize(CIFAR_MEAN,
                                     CIFAR_STD)
    if permute:
        permute_op = RowColPermute()
        transform = transforms.Compose([transforms.ToTensor(), permute_op, normalize])

    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), normalize]
        )

    testset = torchvision.datasets.CIFAR100(
        root=path, train=False, download=True, transform=transform
    )

    return testset

'''
Spherical CIFAR-100 related
'''
def load_scifar100_data(path, val_split=0.2, train=True):

    data_file = os.path.join(path, 's2_cifar100.gz')
    with gzip.open(data_file, 'rb') as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(
        dataset["train"]["images"][:, :, :, :].astype(np.float32))
    train_labels = torch.from_numpy(
        dataset["train"]["labels"].astype(np.int64))


    all_train_dataset = data_utils.TensorDataset(train_data, train_labels)
    print(len(all_train_dataset))
    if val_split == 0.0 or not train:
        val_dataset = None
        train_dataset = all_train_dataset
    else:
        ntrain = int((1-val_split) * len(all_train_dataset))
        train_dataset = data_utils.TensorDataset(train_data[:ntrain], train_labels[:ntrain])
        val_dataset = data_utils.TensorDataset(train_data[ntrain:], train_labels[ntrain:])

    print(len(train_dataset))
    test_data = torch.from_numpy(
        dataset["test"]["images"][:, :, :, :].astype(np.float32))
    test_labels = torch.from_numpy(
        dataset["test"]["labels"].astype(np.int64))

    test_dataset = data_utils.TensorDataset(test_data, test_labels)

    return train_dataset, val_dataset, test_dataset



'''
sEMG Ninapro DB5 related
'''
def load_ninapro_data(path, train=True):

    trainset = load_ninapro(path, 'train')
    valset = load_ninapro(path, 'val')
    testset = load_ninapro(path, 'test')

    if train:
        return trainset, valset, testset

    else:
        trainset = data_utils.ConcatDataset([trainset, valset])

    return trainset, None, testset

def load_ninapro(path, whichset):
    data_str = 'ninapro_' + whichset + '.npy'
    label_str = 'label_' + whichset + '.npy'

    data = np.load(os.path.join(path, data_str),
                             encoding="bytes", allow_pickle=True)
    labels = np.load(os.path.join(path, label_str), encoding="bytes", allow_pickle=True)

    data = np.transpose(data, (0, 2, 1))
    data = data[:, None, :, :]
    print(data.shape)
    print(labels.shape)
    data = torch.from_numpy(data.astype(np.float32))
    labels = torch.from_numpy(labels.astype(np.int64))

    all_data = data_utils.TensorDataset(data, labels)
    return all_data


