from typing import Callable, Optional, Union

import torch
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

@DATAMODULE_REGISTRY
class SphericalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        channels_last: bool = True,
        random_crop: Optional[int] = None,
        data_dir: Optional[str] = ".cache",
        val_split: Union[int, float] = 10_000, # Which split is used elsewhere?
        num_workers: int = 3,
        batch_size: int = 64,
        normalize: bool = True,
        pin_memory: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self._image_shape = [3, 60, 60]
        self.num_classes = 100

        if channels_last:
            self._image_shape = self._image_shape[1], self._image_shape[2], self._image_shape[0]

    def prepare_data(self):
        Spherical(root='../datasets/spherical', train=True, 
                transform=self.default_transforms())
        Spherical(root='../datasets/spherical', train=False, 
                transform=self.default_transforms())

    def setup(self, stage):
        self.spherical_train = Spherical(
                root='../datasets/spherical', 
                train=True, transform=self.default_transforms())
        self.spherical_test = Spherical(
                root='../datasets/spherical', 
                train=False, transform=self.default_transforms())

    def train_dataloader(self):
        spherical_train = DataLoader(self.spherical_train, batch_size=128, shuffle=True, num_workers=8)
        return spherical_train

    def val_dataloader(self):
        spherical_val = DataLoader(self.spherical_test, batch_size=128, shuffle=False, num_workers=8)
        return spherical_val

    def test_dataloader(self):
        return DataLoader(self.spherical_test, batch_size=128, shuffle=False, num_workers=8)

    @property
    def image_shape(self):
        return self._image_shape

    def default_transforms(self) -> Callable:
        return spherical_transform(
            normalize=self.hparams.normalize,
            channels_last=self.hparams.channels_last,
            #random_crop=self.hparams.random_crop,
        )


class Spherical(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        # TODO add automatic downloader...
        data = np.load(f'{root}/s2_cifar100', allow_pickle=True)
        if self.train:
            split = 'train'
        else:
            split = 'test'
        self.x = data[split]['images'].transpose(0, 2, 3, 1)
        self.y = data[split]['labels']

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


def spherical_transform(normalize: bool = True, channels_last: bool = True, random_crop: Optional[int] = None):
    transform_list = []

    def channels_to_last(img: torch.Tensor):
        return img.permute(1, 2, 0).contiguous()

    if random_crop:
        transform_list.append(transforms.RandomCrop(random_crop))

    transform_list.append(transforms.ToTensor())

    if normalize:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    if channels_last:
        transform_list.append(channels_to_last)

    return transforms.Compose(transform_list)
