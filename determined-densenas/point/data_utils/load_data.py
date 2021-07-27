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

from .audio_dataset import *


def load_data(task, path, train=True, permute=False):
    if task == 'smnist':
        return load_smnist_data(path, 0.16, train)

    elif task == 'scifar100':
        return load_scifar100_data(path, 0.2, train)

    elif task == 'sEMG':
        return load_sEMG_data(path, train)

    elif task == 'ninapro':
        return load_ninapro_data(path, train)

    elif task == 'cifar10':
        trainset, valset = load_cifar10_train_data(path, permute, 0.2, train)
        testset = load_cifar10_test_data(path, permute)
        return trainset, valset, testset

    elif task == 'cifar100':
        trainset, valset = load_cifar100_train_data(path, permute, 0.2, train)
        testset = load_cifar100_test_data(path, permute)
        return trainset, valset, testset

    elif task == 'audio':
        return load_audio(path, 'mel', train)

    else:
        raise NotImplementedError


'''
spherical CIFAR100 related
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
spherical MNIST related
'''

def load_smnist_data(path, val_split=0.16, train=True):

    data_file = os.path.join(path, 's2_mnist.gz')
    with gzip.open(data_file, 'rb') as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(
        dataset["train"]["images"][:, None, :, :].astype(np.float32))
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
        dataset["test"]["images"][:, None, :, :].astype(np.float32))
    test_labels = torch.from_numpy(
        dataset["test"]["labels"].astype(np.int64))

    test_dataset = data_utils.TensorDataset(test_data, test_labels)

    return train_dataset, val_dataset, test_dataset

'''
CIFAR related
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




def load_cifar10_train_data(path, permute=False, val_split=0.2, train=True):
    #We could include cutout in transforms
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

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


    all_trainset = torchvision.datasets.CIFAR10(
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

def load_cifar10_test_data(path, permute=False):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    normalize = transforms.Normalize(CIFAR_MEAN,
                                     CIFAR_STD)
    if permute:
        permute_op = RowColPermute()
        transform = transforms.Compose([transforms.ToTensor(), permute_op, normalize])

    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), normalize]
        )

    testset = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform
    )

    return testset


'''
cifar 100
'''
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


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
             normalize, Cutout(16)]
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
sEMG related
'''
def scramble(examples, labels, second_labels=[]):
    random_vec = np.arange(len(labels))
    np.random.shuffle(random_vec)
    new_labels = []
    new_examples = []
    if len(second_labels) == len(labels):
        new_second_labels = []
        for i in random_vec:
            new_labels.append(labels[i])
            new_examples.append(examples[i])
            new_second_labels.append(second_labels[i])
        return new_examples, new_labels, new_second_labels
    else:
        for i in random_vec:
            new_labels.append(labels[i])
            new_examples.append(examples[i])
        return new_examples, new_labels

def load_sEMG_train_data(path):
    datasets_training = np.load(os.path.join(path, "saved_pre_training_dataset_spectrogram.npy"),
            encoding="bytes", allow_pickle=True)
    examples_training, labels_training = datasets_training

    examples_personne_training = []
    labels_gesture_personne_training = []
    for j in range(19):
        print("CURRENT DATASET : ", j)

        for k in range(len(examples_training[j])):
            examples_personne_training.extend(examples_training[j][k])
            labels_gesture_personne_training.extend(labels_training[j][k])

    examples_personne_scrambled, labels_gesture_personne_scrambled = scramble(examples_personne_training,
                                                                                  labels_gesture_personne_training)
    train = data_utils.TensorDataset(torch.from_numpy(np.array(examples_personne_scrambled, dtype=np.float32)),
                              torch.from_numpy(np.array(labels_gesture_personne_scrambled, dtype=np.int64)))


    return train

def load_sEMG_val_data(path):
    datasets_val = np.load(os.path.join(path, "saved_evaluation_dataset_training.npy"),
                                encoding="bytes", allow_pickle=True)
    examples_val, labels_val = datasets_val
    #examples_training = examples_training.reshape(-1, *examples_training.shape[2:])
    #labels_training = labels_training.reshape(-1, *labels_training.shape[2:])

    examples_personne_val = []
    labels_gesture_personne_val = []
    for j in range(17):
        print("CURRENT DATASET : ", j)

        for k in range(len(examples_val[j])):
            examples_personne_val.extend(examples_val[j][k])
            labels_gesture_personne_val.extend(labels_val[j][k])

    examples_personne_scrambled, labels_gesture_personne_scrambled = scramble(examples_personne_val,
                                                                                  labels_gesture_personne_val)
    val = data_utils.TensorDataset(torch.from_numpy(np.array(examples_personne_scrambled, dtype=np.float32)),
                              torch.from_numpy(np.array(labels_gesture_personne_scrambled, dtype=np.int64)))


    return val


def load_sEMG_test_data(path):

    datasets_test0 = np.load(os.path.join(path, "saved_evaluation_dataset_test0.npy"),
                             encoding="bytes", allow_pickle=True)
    examples_test0, labels_test0 = datasets_test0

    datasets_test1 = np.load(os.path.join(path, "saved_evaluation_dataset_test1.npy"),
                             encoding="bytes", allow_pickle=True)
    examples_test1, labels_test1 = datasets_test1

    #x_val = np.concatenate((examples_test0.reshape(-1), examples_test1.reshape(-1)))
    #y_val = np.concatenate((labels_test0.reshape(-1), labels_test1.reshape(-1)))

    X_test_0, Y_test_0 = [], []
    
    X_test_1, Y_test_1 = [], []
    for j in range(17):
        for k in range(len(examples_test0[j])):
            X_test_0.extend(examples_test0[j][k])
            Y_test_0.extend(labels_test0[j][k])

        for k in range(len(examples_test1[j])):
            X_test_1.extend(examples_test1[j][k])
            Y_test_1.extend(labels_test1[j][k])

    X_test_0, Y_test_0 = np.array(X_test_0, dtype=np.float32), np.array(Y_test_0, dtype=np.int64)
    X_test_1, Y_test_1 = np.array(X_test_1, dtype=np.float32), np.array(Y_test_1, dtype=np.int64)
    X_test = np.concatenate((X_test_0, X_test_1))
    Y_test = np.concatenate((Y_test_0, Y_test_1))

    test = data_utils.TensorDataset(torch.from_numpy(X_test),
                  torch.from_numpy(Y_test))

    return test
'''
def load_sEMG_data(path, train=True):
    dataset1, dataset2, dataset3 = load_sEMG_train_data(path), load_sEMG_val_data(path), load_sEMG_test_data(path)
    dataset_list = [dataset1, dataset2, dataset3]
    all_sEMG = data_utils.ConcatDataset(dataset_list)
    total_size = len(all_sEMG)
    print(total_size)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size

    trainset, valset, testset = data_utils.random_split(all_sEMG, [train_size, val_size, test_size])

    return trainset, valset, testset
'''

'''sEMG Myo'''
def load_sEMG_data(path, train):
    train_val_set = torch.load(os.path.join(path, 'trainval_Myo.pt'))
    testset = torch.load(os.path.join(path, 'test_Myo.pt'))

    total_size = len(train_val_set)
    print('sEMG samples: ', total_size)
    train_size = int(total_size * 0.875)
    val_size = total_size - train_size

    if train:
        trainset, valset = data_utils.random_split(train_val_set, [train_size, val_size])
    else:
        trainset, valset = train_val_set, None

    return trainset, valset, testset

'''sEMG ninapro data'''
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

'''
Audio related
'''
def load_audio(path, feature='mel', train=True):
    meta_root = os.path.join(path, "data/chunks")
    train_manifest = "tr.csv"
    val_manifest = "val.csv"
    label_map = "lbl_map.json"
    test_manifest = 'eval.csv'

    train_manifest = os.path.join(meta_root, train_manifest)
    val_manifest = os.path.join(meta_root, val_manifest)
    test_manifest = os.path.join(meta_root, test_manifest)
    label_map = os.path.join(meta_root, label_map)

    bg_files = os.path.join(path, "data/noise_22050")

    if feature == 'mel':
        audio_config = {
            'feature': 'melspectrogram',
            'sample_rate': 22050,
            'min_duration': 1,
            'bg_files': bg_files,
        }
    elif feature == 'raw':
        audio_config = {
            'feature': 'spectrogram',
            'n_fft': 441,
            'hop_len': 220,
            'normalize': False,
            'sample_rate': 22050,
            'min_duration': 1,
            'bg_files': bg_files,
        }
    else:
        raise KeyError

    mixer = BackgroundAddMixer()
    train_mixer = UseMixerWithProb(mixer, 0.75)
    train_transforms = get_transforms_fsd_chunks(True, 101)
    val_transforms = get_transforms_fsd_chunks(False, 101)
    #precision = 16

    train_set = SpectrogramDataset(train_manifest,
                                        label_map,
                                        audio_config,
                                        mode="multilabel", augment=True,
                                        mixer=train_mixer,
                                        transform=train_transforms)

    val_set = FSD50kEvalDataset(val_manifest, label_map,
                                     audio_config,
                                     transform=val_transforms
                                     )

    test_set = FSD50kEvalDataset(test_manifest, label_map,
                                 audio_config,
                                 transform=val_transforms)

    if train:
        return train_set, val_set, test_set

    return train_set, None, test_set