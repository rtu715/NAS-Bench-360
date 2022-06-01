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
from .cosmic_utils import get_dirs, PairedDatasetImagePath


@DATAMODULE_REGISTRY
class CosmicDataModule(pl.LightningDataModule):
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
        self._image_shape = [1, 128, 128]
        self.dense_pred_shape = (1, 128, 128)
        self.batch_size = batch_size
        self.root = root

        if channels_last:
            self._image_shape = self._image_shape[1], self._image_shape[2], self._image_shape[0]

    def prepare_data(self):
        load_cosmic_data(f"{self.root}/cosmic")

    def setup(self, stage):
        self.cosmic_train, self.cosmic_val, self.cosmic_test = load_cosmic_data(f"{self.root}/cosmic")

    def train_dataloader(self):
        dl = DataLoader(self.cosmic_train, batch_size=self.batch_size, shuffle=True, num_workers=0)
        print(len(dl))
        return dl

    def val_dataloader(self):
        dl = DataLoader(self.cosmic_val, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return dl

    def test_dataloader(self):
        dl = DataLoader(self.cosmic_test, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return dl

    @property
    def image_shape(self):
        return self._image_shape

    def default_transforms(self) -> Callable:
        return cosmic_transform()


class ToChannelsLast:
    def __call__(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise RuntimeError
        return x.to(memory_format=torch.channels_last)

    def __repr__(self):
        return self.__class__.__name__ + "()"


def cosmic_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform_list.append(ToChannelsLast())
    return transforms.Compose(transform_list)


def load_cosmic_data(path):

    print(path)

    # get_dirs(path, path) # TODO ?
    train_dirs = np.load(os.path.join(path, "train_dirs.npy"), allow_pickle=True)
    test_dirs = np.load(os.path.join(path, "test_dirs.npy"), allow_pickle=True)

    # Hacky
    if path == "datasets/cosmic":
        # Exclude ../
        train_dirs = [td[3:] for td in train_dirs]
        test_dirs = [td[3:] for td in test_dirs]

    aug_sky = (-0.9, 3)

    # only train f435 and GAL flag for now
    print(train_dirs[0])
    trainvalset_full = PairedDatasetImagePath(train_dirs[::], aug_sky[0], aug_sky[1], part="train")

    testset = PairedDatasetImagePath(train_dirs[::], aug_sky[0], aug_sky[1], part="None")

    train_val_size = len(trainvalset_full)
    val_size = train_val_size // 10

    trainset = Subset(trainvalset_full, np.arange(train_val_size)[:-val_size])
    valset = Subset(trainvalset_full, np.arange(train_val_size)[-val_size:])

    return trainset, valset, testset
