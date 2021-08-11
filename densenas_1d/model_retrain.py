import argparse
import ast
import importlib
import logging
import os
import pprint
import sys
import time
import boto3
from typing import Any, Dict, Sequence, Tuple, Union, cast

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

from determined.pytorch import (
    PyTorchTrial,
    PyTorchTrialContext,
    DataLoader,
    LRScheduler,
    PyTorchCallback
)

from configs.point_train_cfg import cfg as config
from generate_random import generate_arch
from models import model_derived
from tools import utils
from tools.lr_scheduler import get_lr_scheduler
from data_utils.load_data import load_data
from data_utils.download_data import download_from_s3

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class DenseNASTrainTrial(PyTorchTrial):

    def __init__(self, trial_context: PyTorchTrialContext) -> None:
        self.context = trial_context
        self.hparams = AttrDict(trial_context.get_hparams())
        self.last_epoch = 0

        pprint.pformat(config)

        cudnn.benchmark = True
        cudnn.enabled = True

        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.cuda()

        config.net_config, config.net_type = self.hparams.net_config, self.hparams.net_type
        derivedNetwork = getattr(model_derived, '%s_Net' % self.hparams.net_type.upper())

        if self.hparams.net_config == 'random':
            rand_arch = generate_arch(self.hparams.task, self.hparams.net_type)
            model = derivedNetwork(rand_arch, task=self.hparams.task, config=config)
        else:
            model = derivedNetwork(config.net_config, task=self.hparams.task, config=config)


        pprint.pformat("Num params = %.2fMB", utils.count_parameters_in_MB(model))
        self.model = self.context.wrap_model(model)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)/ 1e6
        print('Parameter size in MB: ', total_params)
        optimizer = torch.optim.SGD(
            model.parameters(),
            config.optim.init_lr,
            momentum=config.optim.momentum,
            weight_decay=config.optim.weight_decay
        )
        self.optimizer = self.context.wrap_optimizer(optimizer)

        scheduler = get_lr_scheduler(config, self.optimizer, self.hparams.num_examples, self.context.get_per_slot_batch_size())
        scheduler.last_step = 0
        self.scheduler = self.context.wrap_lr_scheduler(scheduler, step_mode=LRScheduler.StepMode.MANUAL_STEP)
        #scheduler = CosineAnnealingLR(self.optimizer, config.train_params.epochs, config.optim.min_lr)
        #self.scheduler = self.context.wrap_lr_scheduler(scheduler, step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH)
        
        self.config = config
        self.download_directory = self.download_data_from_s3()

    def download_data_from_s3(self):
        '''Download data from s3 to store in temp directory'''

        s3_bucket = self.context.get_data_config()["bucket"]
        download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        s3 = boto3.client("s3")
        os.makedirs(download_directory, exist_ok=True)

        download_from_s3(s3_bucket, self.hparams.task, download_directory)

        if self.hparams.net_config == 'random':
            self.train_data, self.val_data, _ = load_data(self.hparams.task, download_directory, True)

        else:
            self.train_data, _, self.val_data = load_data(self.hparams.task, download_directory, False)

        return download_directory

    def build_training_data_loader(self) -> DataLoader:

        trainset = self.train_data

        return DataLoader(trainset, batch_size=self.context.get_per_slot_batch_size(), shuffle=True, num_workers=2)

    def build_validation_data_loader(self) -> DataLoader:

        valset = self.val_data

        return DataLoader(valset, batch_size=self.context.get_per_slot_batch_size(), shuffle=False, num_workers=2)

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:

        x_train, y_train = batch
        self.scheduler.step()
        logits = self.model(x_train)
        loss = self.criterion(logits, y_train)
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        return {
            'loss': loss,
        }

    def evaluate_full_dataset(
            self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        if self.hparams.task == 'ECG':
            return self.evaluate_full_dataset_ECG(data_loader)

        elif self.hparams.task == 'satellite':
            return self.evaluate_full_dataset_satellite(data_loader)

        return None

    def evaluate_full_dataset_ECG(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:

        obj = utils.AverageMeter()
        all_pred_prob = []

        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                n = input.size(0)
                logits = self.model(input)
                loss = self.criterion(logits, target)
                obj.update(loss, n)
                all_pred_prob.append(logits.cpu().data.numpy())

        '''for ecg validation'''
        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = np.argmax(all_pred_prob, axis=1)

        ## vote most common
        final_pred = []
        final_gt = []
        pid_test = self.val_data.pid
        for i_pid in np.unique(pid_test):
            tmp_pred = all_pred[pid_test == i_pid]
            tmp_gt = self.val_data.label[pid_test == i_pid]
            final_pred.append(Counter(tmp_pred).most_common(1)[0][0])
            final_gt.append(Counter(tmp_gt).most_common(1)[0][0])

        ## classification report
        tmp_report = classification_report(final_gt, final_pred, output_dict=True)
        print(confusion_matrix(final_gt, final_pred))
        f1_score = (tmp_report['0']['f1-score'] + tmp_report['1']['f1-score'] + tmp_report['2']['f1-score'] +
                    tmp_report['3']['f1-score']) / 4

        return {
            'validation_loss': obj.avg,
            'score': f1_score,
        }

    def evaluate_full_dataset_satellite(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:

        obj = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                n = input.size(0)
                logits = self.model(input)
                loss = self.criterion(logits, target)
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                obj.update(loss, n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)
        return {
            'validation_loss': obj.avg,
            'validation_accuracy': top1.avg,
            'validation_top5': top5.avg
        }

