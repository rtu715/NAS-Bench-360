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
from data_utils.audio_dataset import *
from data_utils.audio_dataset import _collate_fn_part, _collate_fn_eval


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

        self.criterion = nn.BCEWithLogitsLoss().cuda()

        config.net_config, config.net_type = self.hparams.net_config, self.hparams.net_type
        derivedNetwork = getattr(model_derived, '%s_Net' % self.hparams.net_type.upper())

        if self.hparams.net_config == 'random':
            self.rand_arch = generate_arch(self.hparams.task, self.hparams.net_type)
            model = derivedNetwork(self.rand_arch, task=self.hparams.task, config=config)

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
        #download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        s3 = boto3.client("s3")
        #os.makedirs(download_directory, exist_ok=True)
        download_directory = '.'
        download_from_s3(s3_bucket, self.hparams.task, download_directory)

        if self.hparams.net_config == 'random':
            self.train_data, self.val_data, _ = load_data(self.hparams.task, download_directory, True, self.hparams.permute)

        else:
            self.train_data, _, self.val_data = load_data(self.hparams.task, download_directory, False, self.hparams.permute)

        return download_directory

    def build_training_data_loader(self) -> DataLoader:

        trainset = self.train_data

        return DataLoader(trainset, num_workers=4, batch_size=self.context.get_per_slot_batch_size(),
                          shuffle=True, sampler=None, collate_fn=_collate_fn_part,
                          pin_memory=False, drop_last=True)

    def build_validation_data_loader(self) -> DataLoader:

        valset = self.val_data

        return DataLoader(valset, sampler=None, num_workers=4,
                          collate_fn=_collate_fn_eval,
                          shuffle=False, batch_size=1,
                          pin_memory=False
                          )

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

    def evaluate_full_dataset(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:

        print(self.rand_arch)

        obj = utils.AverageMeter()
        val_predictions = []
        val_gts = []

        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                n = input.size(0)
                logits = self.model(input)
                logits = logits.mean(0).unsqueeze(0)
                loss = self.criterion(logits, target)
                obj.update(loss, n)
                logits_sigmoid = torch.sigmoid(logits)
                val_predictions.append(logits_sigmoid.detach().cpu().numpy()[0])
                val_gts.append(target.detach().cpu().numpy()[0])

        val_preds = np.asarray(val_predictions).astype('float32')
        val_gts = np.asarray(val_gts).astype('int32')
        map_value = average_precision_score(val_gts, val_preds, average="macro")

        stats = calculate_stats(val_preds, val_gts)
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        

        results = {
            "test_loss": obj.avg,
            "test_mAUC": mAUC,
            "test_mAP": mAP,
            "mAP_value": map_value,
            "dPrime": d_prime(mAUC),

        }

        return results

