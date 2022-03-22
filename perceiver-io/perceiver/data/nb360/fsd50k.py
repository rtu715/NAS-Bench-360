from typing import Callable, Optional, Union

import torch
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets
from .fsd50kutils.load_data import load_audio
from .fsd50kutils.audio_dataset import _collate_fn, _collate_fn_eval

@DATAMODULE_REGISTRY
class FSD50KDataModule(pl.LightningDataModule):
    def __init__(
        self,
        channels_last: bool = True,
        random_crop: Optional[int] = None,
        data_dir: Optional[str] = ".cache",
        num_workers: int = 3,
        batch_size: int = 64,
        normalize: bool = True,
        pin_memory: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self._image_shape = [1, 96, 101]
        self.num_classes = 200
        self.batch_size = batch_size

        if channels_last:
            self._image_shape = self._image_shape[1], self._image_shape[2], self._image_shape[0]

    def prepare_data(self):
        pass
        #load_audio('../datasets/audio', feature='mel', train=True)

    def setup(self, stage):
        self.audio_train, self.audio_val, self.audio_test = load_audio(
            '../datasets/audio', feature='mel', train=True)

    def train_dataloader(self):
        return DataLoader(
            self.audio_train, collate_fn=_collate_fn,
            batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(
            self.audio_val, collate_fn=_collate_fn_eval,
            batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(
            self.audio_test, collate_fn=_collate_fn_eval,
            batch_size=self.batch_size, shuffle=False, num_workers=8)

    @property
    def image_shape(self):
        return self._image_shape

    def default_transforms(self) -> Callable:
        return audio_transform(channels_last=self.hparams.channels_last)

def audio_transform(channels_last: bool = True):
    transform_list = []

    def channels_to_last(img: torch.Tensor):
        return img.permute(1, 2, 0).contiguous()

    transform_list.append(transforms.ToTensor())

    if channels_last:
        transform_list.append(channels_to_last)

    return transforms.Compose(transform_list)
