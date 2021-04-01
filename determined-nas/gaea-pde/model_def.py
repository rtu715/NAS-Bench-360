from typing import Any, Dict, Union, Sequence
import os
from collections import namedtuple
import boto3
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from determined.pytorch import (
    PyTorchTrial,
    PyTorchTrialContext,
    DataLoader,
    LRScheduler,
    PyTorchCallback
)

from data import BilevelDataset
from model_search import Network
from optimizer import EG
from utils import AttrDict, LpLoss, MatReader, UnitGaussianNormalizer
import utils

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class GenotypeCallback(PyTorchCallback):
    def __init__(self, context):
        self.model = context.models[0]

    def on_validation_end(self, metrics):
        print(self.model.genotype())


class GAEASearchTrial(PyTorchTrial):
    def __init__(self, trial_context: PyTorchTrialContext) -> None:
        self.context = trial_context
        self.data_config = trial_context.get_data_config()
        self.hparams = AttrDict(trial_context.get_hparams())
        self.last_epoch = 0

        self.download_directory = self.download_data_from_s3()
        self.grid, self.s = utils.create_grid(self.hparams["sub"])


        # Initialize the models.
        self.criterion = LpLoss(size_average=False)
        self.model = self.context.wrap_model(
            Network(
                self.hparams.init_channels,
                self.hparams.n_classes,
                self.hparams.layers,
                self.criterion,
                self.hparams.nodes,
                width = self.hparams.width,
                k=self.hparams.shuffle_factor,
            )
        )

        # Initialize the optimizers and learning rate scheduler.
        self.ws_opt = self.context.wrap_optimizer(
            torch.optim.SGD(
                self.model.ws_parameters(),
                self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        )
        self.arch_opt = self.context.wrap_optimizer(
            EG(
                self.model.arch_parameters(),
                self.hparams.arch_learning_rate,
                lambda p: p / p.sum(dim=-1, keepdim=True),
            )
        )

        self.lr_scheduler = self.context.wrap_lr_scheduler(
            lr_scheduler=CosineAnnealingLR(
                self.ws_opt,
                self.hparams.scheduler_epochs,
                self.hparams.min_learning_rate,
            ),
            step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH,
        )

    def download_data_from_s3(self):
        '''Download pde data from s3 to store in temp directory'''

        s3_bucket = self.context.get_data_config()["bucket"]
        download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        data_files = ["piececonst_r421_N1024_smooth1.mat", "piececonst_r421_N1024_smooth2.mat"]

        s3 = boto3.client("s3")
        os.makedirs(download_directory, exist_ok=True)

        for data_file in data_files:
            filepath = os.path.join(download_directory, data_file)
            if not os.path.exists(filepath):
                s3.download_file(s3_bucket, data_file, filepath)

        return download_directory

    def build_training_data_loader(self) -> DataLoader:
        """
        For bi-level NAS, we'll need each instance from the dataloader to have one image
        for training shared-weights and another for updating architecture parameters.
        """
        ntrain = 1000
        s = self.s
        r = self.hparams["sub"]

        TRAIN_PATH = os.path.join(self.download_directory, 'piececonst_r421_N1024_smooth1.mat')
        reader = MatReader(TRAIN_PATH)
        x_train = reader.read_field('coeff')[:ntrain, ::r, ::r][:, :s, :s]
        y_train = reader.read_field('sol')[:ntrain, ::r, ::r][:, :s, :s]

        self.x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = self.x_normalizer.encode(x_train)

        self.y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = self.y_normalizer.encode(y_train)

        x_train = torch.cat([x_train.reshape(ntrain, s, s, 1), self.grid.repeat(ntrain, 1, 1, 1)], dim=3)
        train_data = torch.utils.data.TensorDataset(x_train, y_train)

        bilevel_data = BilevelDataset(train_data)

        self.train_data = bilevel_data

        train_queue = DataLoader(
            bilevel_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=2,
        )
        return train_queue

    def build_validation_data_loader(self) -> DataLoader:
        ntest = 100
        s = self.s
        r = self.hparams["sub"]

        TEST_PATH = os.path.join(self.download_directory, 'piececonst_r421_N1024_smooth1.mat')
        reader = MatReader(TEST_PATH)
        x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]
        y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]

        x_test = self.x_normalizer.encode(x_test)
        x_test = torch.cat([x_test.reshape(ntest, s, s, 1), self.grid.repeat(ntest, 1, 1, 1)], dim=3)

        return DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                          batch_size=self.context.get_per_slot_batch_size(), shuffle=False, num_workers=2,)

    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        if epoch_idx != self.last_epoch:
            self.train_data.shuffle_val_inds()
        self.last_epoch = epoch_idx
        x_train, y_train, x_val, y_val = batch
        batch_size = self.context.get_per_slot_batch_size()

        # Train shared-weights
        for a in self.model.arch_parameters():
            a.requires_grad = False
        for w in self.model.ws_parameters():
            w.requires_grad = True

        self.y_normalizer.cuda()
        logits = self.model(x_train)
        target = self.y_normalizer.decode(y_train)
        logits = self.y_normalizer.decode(logits)
        loss = self.criterion(logits.view(batch_size, -1), target.view(batch_size, -1))

        self.context.backward(loss)
        self.context.step_optimizer(self.ws_opt)

        # Train arch parameters
        for a in self.model.arch_parameters():
            a.requires_grad = True
        for w in self.model.ws_parameters():
            w.requires_grad = False

        logits = self.model(x_val)
        target = self.y_normalizer.decode(y_val)
        logits = self.y_normalizer.decode(logits)
        arch_loss = self.criterion(logits.view(batch_size, -1), target.view(batch_size, -1))

        self.context.backward(arch_loss)
        self.context.step_optimizer(self.arch_opt)

        return {
            "loss": loss,
            "arch_loss": arch_loss,
        }

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        input, target = batch
        batch_size = self.context.get_per_slot_batch_size()

        logits = self.model(input)
        logits = self.y_normalizer.decode(logits)

        loss = self.criterion(logits.view(batch_size, -1), target.view(batch_size, -1)).item()
        validation_error = loss / batch_size

        return {"validation_error": validation_error}

    def build_callbacks(self):
        return {"genotype": GenotypeCallback(self.context)}
