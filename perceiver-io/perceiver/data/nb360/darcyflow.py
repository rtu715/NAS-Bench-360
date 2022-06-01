from imghdr import tests
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
from .darcy_utils import MatReader, UnitGaussianNormalizer


@DATAMODULE_REGISTRY
class DarcyFlowDataModule(pl.LightningDataModule):
    def __init__(
        self,
        channels_last: bool = True,
        data_dir: Optional[str] = ".cache",
        num_workers: int = 3,
        batch_size: int = 4,
        pin_memory: bool = False,
        root="../datasets",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._image_shape = [1, 85, 85]
        self.dense_pred_shape = (1, 85, 85)
        self.batch_size = batch_size
        self.root = root

        if channels_last:
            self._image_shape = self._image_shape[1], self._image_shape[2], self._image_shape[0]

    def prepare_data(self):
        load_darcyflow_data(f"{self.root}/darcyflow")

    def setup(self, stage):
        self.darcyflow_train, self.darcyflow_val, self.darcyflow_test, _ = load_darcyflow_data(f"{self.root}/darcyflow")

    def train_dataloader(self):
        dl = DataLoader(self.darcyflow_train, batch_size=self.batch_size, shuffle=True, num_workers=0)
        print(len(dl))
        return dl

    def val_dataloader(self):
        dl = DataLoader(self.darcyflow_val, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return dl

    def test_dataloader(self):
        dl = DataLoader(self.darcyflow_test, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return dl

    @property
    def image_shape(self):
        return self._image_shape

    def default_transforms(self) -> Callable:
        return darcyflow_transform()


def darcyflow_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def load_darcyflow_data(path):
    TRAIN_PATH = f"{path}/piececonst_r421_N1024_smooth1.mat"
    TEST_PATH = f"{path}/piececonst_r421_N1024_smooth2.mat"

    ntrain = 1000
    ntest = 100

    nvalsplit = 100

    r = 5
    s = int(((421 - 1) / r) + 1)

    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field("coeff")[:ntrain, ::r, ::r][:, :s, :s]
    y_train = reader.read_field("sol")[:ntrain, ::r, ::r][:, :s, :s]

    reader.load_file(TEST_PATH)
    x_test = reader.read_field("coeff")[:ntest, ::r, ::r][:, :s, :s]
    y_test = reader.read_field("sol")[:ntest, ::r, ::r][:, :s, :s]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    y_train = y_normalizer.decode(y_train)

    x_train = x_train.reshape(ntrain, s, s, 1)
    x_test = x_test.reshape(ntest, s, s, 1)

    trainset_full = torch.utils.data.TensorDataset(x_train, y_train)
    trainset = Subset(trainset_full, np.arange(ntrain)[:-nvalsplit])
    valset = Subset(trainset_full, np.arange(ntrain)[-nvalsplit:])
    testset = torch.utils.data.TensorDataset(x_test, y_test)

    return trainset, valset, testset, y_normalizer
