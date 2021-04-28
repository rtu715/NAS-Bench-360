"""
This example uses the distributed training aspect of Determined
to quickly and efficiently train a state-of-the-art architecture
for ImageNet found by a leading NAS method called GAEA:
https://arxiv.org/abs/2004.07802

We will add swish activation and squeeze-and-excite modules in this
model to further improve upon the published 24.0 test error on imagenet.

We assume that you already have imagenet downloaded and the train and test
directories set up.
"""

from collections import namedtuple
from typing import Any, Dict

import os
import boto3

import torchvision.transforms as transforms
from torch import nn

from data import ImageNetDataset
from determined.pytorch import (
    DataLoader,
    LRScheduler,
    PyTorchTrial,
    PyTorchTrialContext,
)
from model import Network
from utils import (
    RandAugment,
    Cutout,
    HSwish,
    Swish,
    accuracy,
    AvgrageMeter,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from lr_schedulers import *

import utils

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")


class GAEAEvalTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.data_config = context.get_data_config()
        self.criterion = nn.CrossEntropyLoss()
        self.download_directory = self.download_data_from_s3()

        self.last_epoch_idx = -1

        self.model = self.context.wrap_model(self.build_model_from_config())

        self.optimizer = self.context.wrap_optimizer(
            torch.optim.SGD(
                self.model.parameters(),
                lr=self.context.get_hparam("learning_rate"),
                momentum=self.context.get_hparam("momentum"),
                weight_decay=self.context.get_hparam("weight_decay"),
            )
        )

        self.lr_scheduler = self.context.wrap_lr_scheduler(
            lr_scheduler=CosineAnnealingLR(
                self.optimizer,
                50,
                0,
            ),
            step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH,
        )


    def build_model_from_config(self):
        '''
        genotype = Genotype(
            normal=[
                ("skip_connect", 1),
                ("skip_connect", 0),
                ("sep_conv_3x3", 2),
                ("sep_conv_3x3", 1),
                ("sep_conv_5x5", 2),
                ("sep_conv_3x3", 0),
                ("sep_conv_5x5", 3),
                ("sep_conv_5x5", 2),
            ],
            normal_concat=range(2, 6),
            reduce=[
                ("max_pool_3x3", 1),
                ("sep_conv_3x3", 0),
                ("sep_conv_5x5", 1),
                ("dil_conv_5x5", 2),
                ("sep_conv_3x3", 1),
                ("sep_conv_3x3", 3),
                ("sep_conv_5x5", 1),
                ("max_pool_3x3", 2),
            ],
            reduce_concat=range(2, 6),
        )
        ''' 
        '''
        best model for spherical MNIST
        genotype = Genotype(normal=[('max_pool_3x3', 0),
                                    ('dil_conv_3x3', 1),
                                    ('dil_conv_5x5', 0),
                                    ('sep_conv_3x3', 2),
                                    ('sep_conv_5x5', 1),
                                    ('max_pool_3x3', 3),
                                    ('sep_conv_3x3', 3),
                                    ('sep_conv_3x3', 1)],
                            normal_concat=range(2, 6),
                            reduce=[('skip_connect', 1),
                                    ('skip_connect', 0),
                                    ('sep_conv_5x5', 1),
                                    ('skip_connect', 0),
                                    ('max_pool_3x3', 0),
                                    ('sep_conv_3x3', 1),
                                    ('max_pool_3x3', 4),
                                    ('max_pool_3x3', 3)],
                            reduce_concat=range(2, 6))

        '''
        genotype= Genotype(normal=[('sep_conv_5x5', 0), 
            ('sep_conv_3x3', 1), 
            ('max_pool_3x3', 0), 
            ('sep_conv_5x5', 1), 
            ('dil_conv_3x3', 0), 
            ('dil_conv_3x3', 2), 
            ('sep_conv_3x3', 2), 
            ('sep_conv_3x3', 0)], 
            normal_concat=range(2, 6), 
            reduce=[('skip_connect', 0), 
                ('avg_pool_3x3', 1), 
                ('dil_conv_3x3', 2), 
                ('max_pool_3x3', 0), 
                ('sep_conv_3x3', 2), 
                ('sep_conv_5x5', 0), 
                ('dil_conv_5x5', 3), 
                ('sep_conv_5x5', 2)], 
            reduce_concat=range(2, 6))

        model = Network(
            self.context.get_hparam("init_channels"),
            self.context.get_hparam("num_classes"),
            self.context.get_hparam("layers"),
            genotype,
            in_channels=3 if self.context.get_hparam('task') == 'cifar' else 1,
            drop_path_prob=self.context.get_hparam("drop_path_prob"),
        )

        return model


    def download_data_from_s3(self):
        '''Download data from s3 to store in temp directory'''

        s3_bucket = self.context.get_data_config()["bucket"]
        download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        s3 = boto3.client("s3")
        os.makedirs(download_directory, exist_ok=True)

        if self.context.get_hparam("task") == 'spherical':
            data_files = ["s2_mnist.gz"]
            for data_file in data_files:
                filepath = os.path.join(download_directory, data_file)
                if not os.path.exists(filepath):
                    s3.download_file(s3_bucket, data_file, filepath)

            self.train_data, self.test_data = utils.load_spherical_data(download_directory)

        elif self.context.get_hparam("task") == 'sEMG':
            #data_files = ["saved_evaluation_dataset_test0.npy", "saved_evaluation_dataset_test1.npy",
            #              "saved_evaluation_dataset_training.npy", "saved_pre_training_dataset_spectrogram.npy"]

            data_files = ['trainval_Myo.pt', 'test_Myo.pt']
            for data_file in data_files:
                filepath = os.path.join(download_directory, data_file)
                s3_path = os.path.join('Myo', data_file)
                if not os.path.exists(filepath):
                    s3.download_file(s3_bucket, s3_path, filepath)

            self.train_data, self.test_data = utils.load_sEMG_data(download_directory)

        elif self.context.get_hparam("task") == 'ninapro':
            data_files = ['ninapro_data.npy', 'ninapro_label.npy']
            for data_file in data_files:
                filepath = os.path.join(download_directory, data_file)
                s3_path = os.path.join('ninapro', data_file)
                if not os.path.exists(filepath):
                    s3.download_file(s3_bucket, s3_path, filepath)

            self.train_data, self.test_data = utils.load_ninapro_data(download_directory)

        elif self.context.get_hparam("task") == 'cifar':
            self.train_data, self.test_data = utils.load_cifar_train_data(download_directory, False), \
                                              utils.load_cifar_test_data(download_directory, False)


        else:
            pass

        return download_directory



    def build_training_data_loader(self) -> DataLoader:

        train_data = self.train_data

        train_queue = DataLoader(
            train_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            pin_memory=True,
            num_workers=self.data_config["num_workers_train"],
        )
        return train_queue

    def build_validation_data_loader(self) -> DataLoader:

        valid_data = self.test_data

        valid_queue = DataLoader(
            valid_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=False,
            pin_memory=True,
            num_workers=self.data_config["num_workers_val"],
        )
        return valid_queue

    def train_batch(
        self, batch: Any, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        if batch_idx == 0 or self.last_epoch_idx < epoch_idx:
            current_lr = self.lr_scheduler.get_last_lr()[0]
            print("Epoch: {} lr {}".format(epoch_idx, current_lr))
        self.last_epoch_idx = epoch_idx

        input, target = batch

        logits = self.model(input)
        loss = self.criterion(logits, target)
        top1, top5 = accuracy(logits, target, topk=(1, 5))

        self.context.backward(loss)
        self.context.step_optimizer(
            self.optimizer,
            clip_grads=lambda params: torch.nn.utils.clip_grad_norm_(
                params, self.context.get_hparam("clip_gradients_l2_norm"),
            ),
        )

        return {"loss": loss, "top1_accuracy": top1, "top5_accuracy": top5}

    def evaluate_batch(self, batch: Any) -> Dict[str, Any]:
        input, target = batch
        logits  = self.model(input)
        loss = self.criterion(logits, target)
        top1, top5 = accuracy(logits, target, topk=(1, 5))


        return {
            "loss": loss,
            "top1_accuracy": top1,
            "top5_accuracy": top5,

        }

