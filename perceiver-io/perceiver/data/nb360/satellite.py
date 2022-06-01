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
class SatelliteDataModule(pl.LightningDataModule):
    def __init__(
        self,
        channels_last: bool = True,
        data_dir: Optional[str] = ".cache",
        val_split: Union[int, float] = 100_000,  # TODO
        num_workers: int = 3,
        batch_size: int = 64,
        pin_memory: bool = False,
        root="../datasets/",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._image_shape = [1, 1, 46]  # TODO
        self.num_classes = 24
        self.batch_size = batch_size
        self.val_split = val_split
        self.root = root

        if channels_last:
            self._image_shape = self._image_shape[1], self._image_shape[2], self._image_shape[0]

    def prepare_data(self):
        load_satellite_data(f"{self.root}/satellite")

    def setup(self, stage):
        satellite_train, _, satellite_test = load_satellite_data(f"{self.root}/satellite")
        self.satellite_train = Subset(satellite_train, np.arange(900_000)[: -self.val_split])
        self.satellite_val = Subset(satellite_train, np.arange(900_000)[-self.val_split :])
        self.satellite_test = satellite_test

    def train_dataloader(self):
        return DataLoader(self.satellite_train, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.satellite_val, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.satellite_test, batch_size=self.batch_size, shuffle=False, num_workers=8)

    @property
    def image_shape(self):
        return self._image_shape

    def default_transforms(self) -> Callable:
        return satellite_transform(channels_last=self.hparams.channels_last)


def satellite_transform(channels_last: bool = True):
    transform_list = []

    def channels_to_last(img: torch.Tensor):
        return img.permute(1, 2, 0).contiguous()

    transform_list.append(transforms.ToTensor())

    if channels_last:
        transform_list.append(channels_to_last)

    return transforms.Compose(transform_list)


def load_satellite_data(path):
    train_file = os.path.join(path, "satellite_train.npy")
    test_file = os.path.join(path, "satellite_test.npy")

    all_train_data, all_train_labels = (
        np.load(train_file, allow_pickle=True)[()]["data"],
        np.load(train_file, allow_pickle=True)[()]["label"],
    )
    test_data, test_labels = (
        np.load(test_file, allow_pickle=True)[()]["data"],
        np.load(test_file, allow_pickle=True)[()]["label"],
    )

    # rerange labels to 0-23
    all_train_labels = all_train_labels - 1
    test_labels = test_labels - 1

    # normalize data
    all_train_data = (all_train_data - all_train_data.mean(axis=1, keepdims=True)) / all_train_data.std(
        axis=1, keepdims=True
    )
    test_data = (test_data - test_data.mean(axis=1, keepdims=True)) / test_data.std(axis=1, keepdims=True)

    all_train_data = all_train_data[:, np.newaxis, :, np.newaxis]
    test_data = test_data[:, np.newaxis, :, np.newaxis]

    # convert to tensor/longtensor
    all_train_tensors, all_train_labeltensor = torch.from_numpy(all_train_data).type(
        torch.FloatTensor
    ), torch.from_numpy(all_train_labels).type(torch.LongTensor)

    test_tensors, test_labeltensor = torch.from_numpy(test_data).type(torch.FloatTensor), torch.from_numpy(
        test_labels
    ).type(torch.LongTensor)
    testset = TensorDataset(test_tensors, test_labeltensor)
    trainset = TensorDataset(all_train_tensors, all_train_labeltensor)

    return trainset, None, testset
