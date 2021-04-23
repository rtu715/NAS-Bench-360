import os
import numpy as np
import torch

import gzip
import pickle
from torch import nn

import torch.utils.data as data_utils

import torchvision
from torchvision import transforms

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def data_transforms_imagenet():
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, valid_transform



def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


'''
spherical MNIST related
'''

def load_spherical_data(path, val_split=0.16):


    data_file = os.path.join(path, 's2_mnist.gz')
    with gzip.open(data_file, 'rb') as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(
        dataset["train"]["images"][:, None, :, :].astype(np.float32))
    train_labels = torch.from_numpy(
        dataset["train"]["labels"].astype(np.int64))


    all_train_dataset = data_utils.TensorDataset(train_data, train_labels)
    print(len(all_train_dataset))
    if val_split == 0.0:
        val_dataset = None
        train_dataset = all_train_dataset
    else:
        ntrain = int((1-val_split) * len(all_train_dataset))
        train_dataset = data_utils.TensorDataset(train_data[:ntrain], train_labels[:ntrain])
        val_dataset = data_utils.TensorDataset(train_data[ntrain:], train_labels[ntrain:])
        
    #train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_data = torch.from_numpy(
        dataset["test"]["images"][:, None, :, :].astype(np.float32))
    test_labels = torch.from_numpy(
        dataset["test"]["labels"].astype(np.int64))

    test_dataset = data_utils.TensorDataset(test_data, test_labels)
    #test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataset, val_dataset, test_dataset

'''
CIFAR related
'''

class RowColPermute(nn.Module):

    def __init__(self, row, col):
        super().__init__()
        self.rowperm = torch.randperm(row) if type(row) == int else row
        self.colperm = torch.randperm(col) if type(col) == int else col

    def forward(self, tensor):
        return tensor[:, self.rowperm][:, :, self.colperm]




def load_cifar_train_data(path, permute, val_split=0.2):
    #We could include cutout in transforms
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    normalize = transforms.Normalize(CIFAR_MEAN,
                                     CIFAR_STD)

    if permute:
        permute_op = RowColPermute(32, 32)
        transform = transforms.Compose([transforms.ToTensor(), permute_op, normalize])

    else:
        transform = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             normalize]
        )


    all_trainset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform
    )

    if val_split==0.0:
        return all_trainset, None
    
    n_train = int((1-val_split) * len(all_trainset))
    train_ind = torch.arange(n_train)
    val_ind = torch.arange(n_train, len(all_trainset))
    trainset = data_utils.Subset(all_trainset, train_ind)
    valset = data_utils.Subset(all_trainset, val_ind)

    return trainset, valset

def load_cifar_test_data(path, permute=False):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    normalize = transforms.Normalize(CIFAR_MEAN,
                                     CIFAR_STD)
    if permute:
        permute_op = RowColPermute(32, 32)
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

    for j in range(19):
        print("CURRENT DATASET : ", j)
        examples_personne_training = []
        labels_gesture_personne_training = []

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

    for j in range(17):
        print("CURRENT DATASET : ", j)
        examples_personne_val = []
        labels_gesture_personne_val = []

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

    for j in range(17):
        X_test_0, Y_test_0 = [], []
        for k in range(len(examples_test0)):
            X_test_0.extend(examples_test0[j][k])
            Y_test_0.extend(labels_test0[j][k])

        X_test_1, Y_test_1 = [], []
        for k in range(len(examples_test1)):
            X_test_1.extend(examples_test1[j][k])
            Y_test_1.extend(labels_test1[j][k])

    X_test_0, Y_test_0 = np.array(X_test_0, dtype=np.float32), np.array(Y_test_0, dtype=np.int64)
    X_test_1, Y_test_1 = np.array(X_test_1, dtype=np.float32), np.array(Y_test_1, dtype=np.int64)
    X_test = np.concatenate((X_test_0, X_test_1))
    Y_test = np.concatenate((Y_test_0, Y_test_1))

    test = data_utils.TensorDataset(torch.from_numpy(X_test),
                  torch.from_numpy(Y_test))

    return test

def load_sEMG_data(path):
    '''merge all sEMG data and apply 80/10/10 partition'''
    '''
    datasets_1 = np.load(os.path.join(path, "saved_pre_training_dataset_spectrogram.npy"),
            encoding="bytes", allow_pickle=True)
    examples_1, labels_1 = datasets_1

    datasets_2 = np.load(os.path.join(path, "saved_evaluation_dataset_training.npy"),
                                encoding="bytes", allow_pickle=True)
    examples_2, labels_2 = datasets_2

    datasets_3 = np.load(os.path.join(path, "saved_evaluation_dataset_test0.npy"),
                             encoding="bytes", allow_pickle=True)
    examples_3, labels_3 = datasets_3

    datasets_4 = np.load(os.path.join(path, "saved_evaluation_dataset_test1.npy"),
                             encoding="bytes", allow_pickle=True)
    examples_4, labels_4 = datasets_4
    '''
    dataset1, dataset2, dataset3 = load_sEMG_train_data(path), load_sEMG_val_data(path), load_sEMG_test_data(path)
    dataset_list = [dataset1, dataset2, dataset3]
    all_sEMG = data_utils.ConcatDataset(dataset_list)
    total_size = len(all_sEMG)

    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size

    trainset, valset, testset = data_utils.random_split(all_sEMG, [train_size, val_size, test_size])

    return trainset, valset, testset