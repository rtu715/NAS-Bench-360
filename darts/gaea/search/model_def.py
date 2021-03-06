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

from data_utils.load_data import load_data
from data_utils.download_data import download_from_s3



TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class GenotypeCallback(PyTorchCallback):
    def __init__(self, context):
        self.model = context.models[0]

    def on_validation_end(self, metrics):
        print(self.model.genotype())


class GAEASearchTrial(PyTorchTrial):
    def __init__(self, trial_context: PyTorchTrialContext) -> None:
        self.context = trial_context
        self.hparams = utils.AttrDict(trial_context.get_hparams())
        self.last_epoch = 0
        self.download_directory = self.download_data_from_s3()

        dataset_hypers = {'sEMG': (7, 1), 'ninapro': (18, 1), 'cifar10': (10, 3), 'smnist': (10, 1), 'cifar100':(100, 3), 'scifar100': (100, 3)}
        n_classes, in_channels = dataset_hypers[self.hparams.task]


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
                in_channels = in_channels,
            )
        )

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)/ 1e6
        print('Parameter size in MB: ', total_params)
        
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

        download_from_s3(s3_bucket, self.hparams.task, download_directory)

        self.train_data, self.val_data, self.test_data = load_data(self.hparams.task, download_directory, True, self.hparams.permute)
        self.build_test_data_loader(download_directory)

        return download_directory

    def build_training_data_loader(self) -> DataLoader:

        trainset = self.train_data
        bilevel = BilevelDataset(trainset)
        self.train_data = bilevel
        print('Length of bilevel dataset: ', len(bilevel))

        return DataLoader(bilevel, batch_size=self.context.get_per_slot_batch_size(), shuffle=True, num_workers=2)

    def build_validation_data_loader(self) -> DataLoader:
        valset = self.val_data
        return DataLoader(valset, batch_size=self.context.get_per_slot_batch_size(), shuffle=False, num_workers=2)

    def build_test_data_loader(self, download_directory):
        
        testset = self.test_data

        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=self.context.get_per_slot_batch_size(), shuffle=False, num_workers=2)
        return 

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
        self.context.step_optimizer(
            optimizer=self.ws_opt,
            clip_grads=lambda params: torch.nn.utils.clip_grad_norm_(
                params,
                self.context.get_hparam("clip_gradients_l2_norm"),
            ),
        )

        arch_loss = 0.0
        # Train arch parameters
        if epoch_idx > 10:
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

    '''
    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        input, target = batch
        logits = self.model(input)
        loss = self.model._loss(input, target)
        top1, top5 = utils.accuracy(logits, target, topk=(1, 5))

        test_input, test_target = next(iter(self.test_loader))
        test_input, test_target = test_input.cuda(), test_target.cuda()
        test_logits = self.model(test_input)
        test_loss = self.model._loss(test_input, test_target)
        test_top1, test_top5 = utils.accuracy(test_logits, test_target, topk=(1, 5))

        return {"loss": loss, "top1_accuracy": top1, "top5_accuracy": top5, "test_loss": test_loss,
                "top1_accuracy_test": test_top1, "top5_accuracy_test": test_top5}
    '''
    def evaluate_full_dataset(
        self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        acc_top1 = utils.AverageMeter()
        acc_top5 = utils.AverageMeter()
        loss_avg = utils.AverageMeter()

        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                n = input.size(0)
                logits = self.model(input)
                loss = self.model._loss(input, target)
                top1, top5 = utils.accuracy(logits, target, topk=(1, 5))
                acc_top1.update(top1.item(), n)
                acc_top5.update(top5.item(), n)
                loss_avg.update(loss, n)
        results = {
            "loss": loss_avg.avg,
            "top1_accuracy": acc_top1.avg,
            "top5_accuracy": acc_top5.avg,
        }


        acc_top1 = utils.AverageMeter()
        acc_top5 = utils.AverageMeter()
        loss_avg = utils.AverageMeter()
        with torch.no_grad():
            for batch in self.test_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                n = input.size(0)
                logits = self.model(input)
                loss = self.model._loss(input, target)
                top1, top5 = utils.accuracy(logits, target, topk=(1, 5))
                acc_top1.update(top1.item(), n)
                acc_top5.update(top5.item(), n)
                loss_avg.update(loss, n)
        results2 = {
            "test_loss": loss_avg.avg,
            "test_top1_accuracy": acc_top1.avg,
            "test_top5_accuracy": acc_top5.avg,
        }

        results.update(results2)

        return results

    def build_callbacks(self):
        return {"genotype": GenotypeCallback(self.context)}
