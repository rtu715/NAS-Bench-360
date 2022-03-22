from typing import Callable, Optional, Union

import torch
import pytorch_lightning as pl
import numpy as np
import pickle
import os
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from collections import Counter
from sklearn.model_selection import train_test_split

@DATAMODULE_REGISTRY
class DeepSEADataModule(pl.LightningDataModule):
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
        self._image_shape = [4, 1, 1000] # TODO
        self.num_classes = 36 # TODO
        self.batch_size = batch_size

        if channels_last:
            self._image_shape = self._image_shape[1], self._image_shape[2], self._image_shape[0]

    def prepare_data(self):
        load_deepsea_data('../datasets/deepsea')

    def setup(self, stage):
        self.deepsea_train, self.deepsea_val, self.deepsea_test = \
            load_deepsea_data('../datasets/deepsea')

    def train_dataloader(self):
        return DataLoader(
            self.deepsea_train, 
            batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(
            self.deepsea_val, 
            batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(
            self.deepsea_test, 
            batch_size=self.batch_size, shuffle=False, num_workers=8)

    @property
    def image_shape(self):
        return self._image_shape

    def default_transforms(self) -> Callable:
        return deepsea_transform(channels_last=self.hparams.channels_last)

def deepsea_transform(channels_last: bool = True):
    transform_list = []

    def channels_to_last(img: torch.Tensor):
        return img.permute(1, 2, 0).contiguous()

    transform_list.append(transforms.ToTensor())

    if channels_last:
        transform_list.append(channels_to_last)

    return transforms.Compose(transform_list)

def load_deepsea_data(path):
    data = np.load(os.path.join(path, 'deepsea_filtered.npz'))
    train_data, train_labels = torch.from_numpy(data['x_train']).type(torch.FloatTensor), \
                                           torch.from_numpy(data['y_train']).type(torch.FloatTensor)
    train_data = train_data[:, np.newaxis, :, :]
    trainset = TensorDataset(train_data, train_labels)

    val_data, val_labels = torch.from_numpy(data['x_val']).type(torch.FloatTensor), \
            torch.from_numpy(data['y_val']).type(torch.FloatTensor)
    val_data = val_data[:, np.newaxis, :, :]
    valset = TensorDataset(val_data, val_labels)

    test_data, test_labels = torch.from_numpy(data['x_test']).type(torch.FloatTensor), \
            torch.from_numpy(data['y_test']).type(torch.FloatTensor)
    test_data = test_data[:, np.newaxis, :, :]
    testset = TensorDataset(test_data, test_labels)

    return trainset, valset, testset
