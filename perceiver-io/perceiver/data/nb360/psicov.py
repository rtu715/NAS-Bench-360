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

from .protein_io_utils import load_list
from .protein_gen_utils import PDNetDataset


@DATAMODULE_REGISTRY
class PSICOVDataModule(pl.LightningDataModule):
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
        self._image_shape = [57, 128, 128]
        self.dense_pred_shape = (1, 128, 128)
        self.batch_size = batch_size
        self.root = root

        if channels_last:
            self._image_shape = self._image_shape[1], self._image_shape[2], self._image_shape[0]

    def prepare_data(self):
        load_psicov_data(f"{self.root}/psicov", self.batch_size)

    def setup(self, stage):
        self.psicov_train, self.psicov_val, self.psicov_test, _, _ = load_psicov_data(
            f"{self.root}/psicov", self.batch_size
        )

    def train_dataloader(self):
        dl = DataLoader(self.psicov_train, batch_size=self.batch_size, shuffle=True, num_workers=0)
        print(len(dl))
        return dl

    def val_dataloader(self):
        dl = DataLoader(self.psicov_val, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return dl

    def test_dataloader(self):
        dl = DataLoader(self.psicov_test, batch_size=1, shuffle=False, num_workers=0)
        return dl

    @property
    def image_shape(self):
        return self._image_shape

    def default_transforms(self) -> Callable:
        return psicov_transform()


def psicov_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def load_psicov_data(path, batch_size):
    all_feat_paths = [f"{path}/deepcov/features/", f"{path}/psicov/features/", f"{path}/cameo/features/"]
    all_dist_paths = [f"{path}/deepcov/distance/", f"{path}/psicov/distance/", f"{path}/cameo/distance/"]

    deepcov_list = load_list(f"{path}/deepcov.lst", -1)
    length_dict = {}
    for pdb in deepcov_list:
        (ly, seqy, cb_map) = np.load(f"{path}/deepcov/distance/" + pdb + "-cb.npy", allow_pickle=True)
        length_dict[pdb] = ly

    # Training set
    print(len(deepcov_list))
    train_pdbs = deepcov_list[100:]
    trainset = PDNetDataset(
        train_pdbs, all_feat_paths, all_dist_paths, 128, 10, batch_size, 57, label_engineering="16.0"
    )

    # Validation set
    valid_pdbs = deepcov_list[:100]
    validset = PDNetDataset(
        valid_pdbs, all_feat_paths, all_dist_paths, 128, 10, batch_size, 57, label_engineering="16.0"
    )

    # Test set (weird)
    psicov_list = load_list(f"{path}/psicov.lst")
    psicov_length_dict = {}
    for pdb in psicov_list:
        (ly, seqy, cb_map) = np.load(f"{path}/psicov/distance/" + pdb + "-cb.npy", allow_pickle=True)
        psicov_length_dict[pdb] = ly
    my_list = psicov_list
    length_dict = psicov_length_dict

    # note, when testing batch size should be different
    testset = PDNetDataset(my_list, all_feat_paths, all_dist_paths, 512, 10, 1, 57, label_engineering=None)

    return trainset, validset, testset, my_list, length_dict
