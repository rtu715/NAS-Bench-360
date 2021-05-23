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
from model import AuxNetworkCIFAR

from utils import (
    RandAugment,
    Cutout,
    HSwish,
    Swish,
    accuracy,
    AverageMeter,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from lr_schedulers import *

import utils

from data_utils.load_data import load_data
from data_utils.download_data import download_from_s3
from searched_genotypes import genotypes

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")


class GAEAEvalTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.data_config = context.get_data_config()
        self.criterion = nn.CrossEntropyLoss()
        self.download_directory = self.download_data_from_s3()

        self.last_epoch_idx = -1

        self.model = self.context.wrap_model(self.build_model_from_config())

        #total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)/ 1e6
        #print('Parameter size in MB: ', total_params)
        print("param size = %f MB" % utils.count_parameters_in_MB(self.model))
        
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
                600.0,
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

        best genotype for ninapro
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
        '''
        #genotype=Genotype(normal=[('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 3), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6)) 
        if self.context.get_hparam('permute'):
            genotype = genotypes['cifar100_permuted']
        else:
            genotype = genotypes[self.context.get_hparam('task')] 
        
        print(self.context.get_hparam('task'))
        print(genotype)

        dataset_hypers = {'sEMG': (7, 1), 'ninapro': (18, 1), 'cifar10': (10, 3), 'smnist': (10, 1), 'cifar100': (100, 3), 'scifar100': (100, 3)}
        n_classes, in_channels = dataset_hypers[self.context.get_hparam('task')]
        
        model = Network(
            self.context.get_hparam("init_channels"),
            n_classes,
            self.context.get_hparam("layers"),
            genotype,
            in_channels=in_channels,
            drop_path_prob=self.context.get_hparam("drop_path_prob"),
        )

        return model

    def download_data_from_s3(self):
        '''Download data from s3 to store in temp directory'''

        s3_bucket = self.context.get_data_config()["bucket"]
        download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        s3 = boto3.client("s3")
        os.makedirs(download_directory, exist_ok=True)

        download_from_s3(s3_bucket, self.context.get_hparam('task'), download_directory)

        self.train_data, _ , self.val_data = load_data(self.context.get_hparam('task'), download_directory, False, permute=self.context.get_hparam('permute'))

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

        valid_data = self.val_data

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
       
            self.model.drop_path_prob = self.context.get_hparam("drop_path_prob") * epoch_idx / 600.0
            print('current drop prob is {}'.format(self.model.drop_path_prob))
        
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


    '''
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
                loss = self.criterion(logits, target)
                top1, top5 = utils.accuracy(logits, target, topk=(1, 5))
                acc_top1.update(top1.item(), n)
                acc_top5.update(top5.item(), n)
                loss_avg.update(loss, n)
        results = {
            "loss": loss_avg.avg,
            "top1_accuracy": acc_top1.avg,
            "top5_accuracy": acc_top5.avg,
        }

        return results

