from typing import Any, Dict, Union, Sequence
import boto3
import os
import tempfile

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
        self.hparams = utils.AttrDict(trial_context.get_hparams())
        self.last_epoch = 0


        self.download_directory = tempfile.mkdtemp()

        if self.hparams.task == 'spherical':
            path = '/workspace/tasks/spherical/s2_mnist.gz'
            self.train_data, self.test_data = utils.load_spherical_data(path, self.context.get_per_slot_batch_size())

        if self.hparams.task == 'sEMG':
            self.download_directory = '/workspace/tasks/MyoArmbandDataset/PyTorchImplementation/sEMG'

        #self.download_directory = self.download_data_from_s3()

        n_classes = 7 if self.hparams['task'] == 'sEMG' else 10
        in_channels = 3 if self.hparams['task'] == 'cifar' else 1

        # Initialize the models.
        criterion = nn.CrossEntropyLoss()
        self.model = self.context.wrap_model(
            Network(
                self.hparams.init_channels,
                n_classes,
                self.hparams.layers,
                criterion,
                self.hparams.nodes,
                k=self.hparams.shuffle_factor,
                in_challes = in_channels,
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
        '''Download data from s3 to store in temp directory'''

        s3_bucket = self.context.get_data_config()["bucket"]
        download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        s3 = boto3.client("s3")
        os.makedirs(download_directory, exist_ok=True)

        if self.hparams.task == 'spherical':
            data_files = ["s2_mnist.gz"]
            for data_file in data_files:
                filepath = os.path.join(download_directory, data_file)
                if not os.path.exists(filepath):
                    s3.download_file(s3_bucket, data_file, filepath)

            self.train_data, self.test_data = utils.load_spherical_data(download_directory,
                                                                           self.context.get_per_slot_batch_size())

        if self.hparams.task == 'sEMG':
            data_files = ["saved_evaluation_dataset_test0.npy", "saved_evaluation_dataset_test1.npy",
                          "saved_evaluation_dataset_training.npy"]
            for data_file in data_files:
                filepath = os.path.join(download_directory, data_file)
                if not os.path.exists(filepath):
                    s3.download_file(s3_bucket, data_file, filepath)

        return download_directory

    def build_training_data_loader(self) -> DataLoader:
        if self.hparams['task'] == 'cifar':
            trainset = utils.load_cifar_train_data(self.download_directory, self.hparams['permute'])

        elif self.hparams['task'] == 'spherical':
            trainset = self.train_data

        elif self.hparams['task'] == 'sEMG':
            trainset = utils.load_sEMG_train_data(self.download_directory)

        else:
            pass

        bilevel = BilevelDataset(trainset)


        return DataLoader(bilevel, batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self) -> DataLoader:

        if self.hparams['task'] == 'cifar':
            valset = utils.load_cifar_val_data(self.download_directory, self.hparams['permute'])

        elif self.hparams['task'] == 'spherical':
            valset = self.test_data

        elif self.hparams['task'] == 'sEMG':
            valset = utils.load_sEMG_val_data(self.download_directory)

        else:
            pass

        return DataLoader(valset, batch_size=self.context.get_per_slot_batch_size())

    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        if epoch_idx != self.last_epoch:
            self.train_data.shuffle_val_inds()
        self.last_epoch = epoch_idx
        x_train, y_train, x_val, y_val = batch

        # Train shared-weights
        for a in self.model.arch_parameters():
            a.requires_grad = False
        for w in self.model.ws_parameters():
            w.requires_grad = True
        loss = self.model._loss(x_train, y_train)
        self.context.backward(loss)
        self.context.step_optimizer(self.ws_opt)

        # Train arch parameters
        for a in self.model.arch_parameters():
            a.requires_grad = True
        for w in self.model.ws_parameters():
            w.requires_grad = False
        arch_loss = self.model._loss(x_val, y_val)
        self.context.backward(arch_loss)
        self.context.step_optimizer(self.arch_opt)

        return {
            "loss": loss,
            "arch_loss": arch_loss,
        }

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        input, target = batch
        logits = self.model(input)
        loss = self.model._loss(input, target)
        top1, top5 = utils.accuracy(logits, target, topk=(1, 5))

        return {"loss": loss, "top1_accuracy": top1, "top5_accuracy": top5}

    def build_callbacks(self):
        return {"genotype": GenotypeCallback(self.context)}
