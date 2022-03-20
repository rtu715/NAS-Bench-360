from typing import Callable, Optional, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

@DATAMODULE_REGISTRY
class CIFAR100DataModule(pl.LightningDataModule):
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
        self._image_shape = [3, 32, 32]
        self.num_classes = 100

        if channels_last:
            self._image_shape = self._image_shape[1], self._image_shape[2], self._image_shape[0]

    def prepare_data(self):
        datasets.CIFAR100(root='../datasets/cifar-100',train=True,download=True, transform=self.default_transforms())
        datasets.CIFAR100(root='../datasets/cifar-100',train=False,download=True, transform=self.default_transforms())

    def setup(self, stage):
        self.cifar_train = datasets.CIFAR100(
                root='../datasets/cifar-100', 
                train=True, download=True, transform=self.default_transforms())
        self.cifar_test = datasets.CIFAR100(
                root='../datasets/cifar-100', 
                train=False, download=True, transform=self.default_transforms())

    def train_dataloader(self):
        cifar_train = DataLoader(self.cifar_train, batch_size=128, shuffle=True, num_workers=8)
        return cifar_train

    def val_dataloader(self):
        cifar_val = DataLoader(self.cifar_test, batch_size=128, shuffle=False, num_workers=8)
        return cifar_val # TODO

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=128, shuffle=False, num_workers=8)

    @property
    def image_shape(self):
        return self._image_shape

    def default_transforms(self) -> Callable:
        return cifar100_transform(
            normalize=self.hparams.normalize,
            channels_last=self.hparams.channels_last,
            random_crop=self.hparams.random_crop,
        )

def cifar100_transform(normalize: bool = True, channels_last: bool = True, random_crop: Optional[int] = None):
    transform_list = []

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

def channels_to_last(img: torch.Tensor):
    return img.permute(1, 2, 0).contiguous()
