from typing import Callable, Optional, Union

import torch
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets


@DATAMODULE_REGISTRY
class NinaProDataModule(pl.LightningDataModule):
    def __init__(
        self,
        channels_last: bool = True,
        random_crop: Optional[int] = None,
        data_dir: Optional[str] = ".cache",
        num_workers: int = 3,
        batch_size: int = 64,
        normalize: bool = True,
        pin_memory: bool = False,
        root="../datasets",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._image_shape = [1, 52, 16]
        self.num_classes = 18
        self.batch_size = batch_size
        self.root = root

        if channels_last:
            self._image_shape = self._image_shape[1], self._image_shape[2], self._image_shape[0]

    def prepare_data(self):
        NinaPro(root=f"{self.root}/ninapro", split="train", transform=self.default_transforms())
        NinaPro(root=f"{self.root}/ninapro", split="val", transform=self.default_transforms())
        NinaPro(root=f"{self.root}/ninapro", split="test", transform=self.default_transforms())

    def setup(self, stage):
        self.ninapro_train = NinaPro(root=f"{self.root}/ninapro", split="train", transform=self.default_transforms())
        self.ninapro_valid = NinaPro(root=f"{self.root}/ninapro", split="val", transform=self.default_transforms())
        self.ninapro_test = NinaPro(root=f"{self.root}/ninapro", split="test", transform=self.default_transforms())

    def train_dataloader(self):
        return DataLoader(self.ninapro_train, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.ninapro_valid, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.ninapro_test, batch_size=self.batch_size, shuffle=False, num_workers=8)

    @property
    def image_shape(self):
        return self._image_shape

    def default_transforms(self) -> Callable:
        return ninapro_transform(channels_last=self.hparams.channels_last)


class NinaPro(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        # TODO add automatic downloader...
        self.x = np.load(f"{root}/ninapro_{split}.npy").astype(np.float32)
        self.x = self.x[:, np.newaxis, :, :].transpose(0, 2, 3, 1)
        self.y = np.load(f"{root}/label_{split}.npy").astype(int)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.x[idx, :]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)
        return x, y


def ninapro_transform(channels_last: bool = True):
    transform_list = []

    def channels_to_last(img: torch.Tensor):
        return img.permute(1, 2, 0).contiguous()

    transform_list.append(transforms.ToTensor())

    if channels_last:
        transform_list.append(channels_to_last)

    return transforms.Compose(transform_list)
