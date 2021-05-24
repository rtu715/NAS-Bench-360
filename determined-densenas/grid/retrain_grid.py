import os
import pprint
import sys
import boto3
import json
from typing import Any, Dict, Sequence, Union

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from determined.pytorch import (
    PyTorchTrial,
    PyTorchTrialContext,
    DataLoader,
    LRScheduler
)

from configs.grid_train_cfg import cfg as config
from models import model_derived
from tools import utils
from utils_grid import LpLoss, MatReader, UnitGaussianNormalizer, LogCoshLoss
from utils_grid import create_grid, filter_MAE

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

        if self.hparams.task == 'pde':
            self.grid, self.s = create_grid(self.hparams.sub)
            self.criterion = LpLoss(size_average=False)
            self.in_channels = 3

        elif self.hparams.task == 'protein':
            self.criterion = LogCoshLoss()
            # error is reported via MAE
            self.error = nn.L1Loss(reduction='sum')
            self.in_channels = 57

        else:
            raise NotImplementedError

        cudnn.benchmark = True
        cudnn.enabled = True

        config.net_config, config.net_type = self.hparams.net_config, self.hparams.net_type
        derivedNetwork = getattr(model_derived, '%s_Net' % self.hparams.net_type.upper())
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

        # scheduler = get_lr_scheduler(config, self.weight_optimizer, self.hparams.num_examples)
        # scheduler.last_step = 0
        scheduler = CosineAnnealingLR(self.optimizer, config.train_params.epochs, config.optim.min_lr)
        self.scheduler = self.context.wrap_lr_scheduler(scheduler, step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH)

        self.config = config
        self.download_directory = self.download_data_from_s3()

    def download_data_from_s3(self):
        '''Download pde data from s3 to store in temp directory'''

        s3_bucket = self.context.get_data_config()["bucket"]
        download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"

        if self.hparams.task == 'pde':
            data_files = ["piececonst_r421_N1024_smooth1.mat", "piececonst_r421_N1024_smooth2.mat"]
            s3_path = None

        elif self.hparams.task == 'protein':
            data_files = ['X_train.npz', 'X_valid.npz', 'Y_train.npz',
                          'Y_valid.npz', 'X_test.npz', 'Y_test.npz', 'psicov.json']
            s3_path = 'protein'

        else:
            raise NotImplementedError

        s3 = boto3.client("s3")
        os.makedirs(download_directory, exist_ok=True)

        for data_file in data_files:
            filepath = os.path.join(download_directory, data_file)
            s3_loc = os.path.join(s3_path, data_file) if s3_path is not None else data_file
            if not os.path.exists(filepath):
                s3.download_file(s3_bucket, s3_loc, filepath)

        return download_directory

    def build_training_data_loader(self) -> DataLoader:
        if self.hparams.task == 'pde':
            TRAIN_PATH = os.path.join(self.download_directory, 'piececonst_r421_N1024_smooth1.mat')
            self.reader = MatReader(TRAIN_PATH)
            s = self.s
            r = self.hparams["sub"]
            ntrain = 1000
            ntest = 100
            x_train = self.reader.read_field('coeff')[:ntrain, ::r, ::r][:, :s, :s]
            y_train = self.reader.read_field('sol')[:ntrain, ::r, ::r][:, :s, :s]

            self.x_normalizer = UnitGaussianNormalizer(x_train)
            x_train = self.x_normalizer.encode(x_train)

            self.y_normalizer = UnitGaussianNormalizer(y_train)
            y_train = self.y_normalizer.encode(y_train)

            x_train = torch.cat([x_train.reshape(ntrain, s, s, 1), self.grid.repeat(ntrain, 1, 1, 1)], dim=3)
            train_data = torch.utils.data.TensorDataset(x_train, y_train)

        elif self.hparams.task == 'protein':
            os.chdir(self.download_directory)
            x_train = np.load('X_train.npz')
            y_train = np.load('Y_train.npz')
            x_train = torch.from_numpy(x_train.f.arr_0)
            y_train = torch.from_numpy(y_train.f.arr_0)

            x_val = np.load('X_valid.npz')
            y_val = np.load('Y_valid.npz')
            x_val = torch.from_numpy(x_val.f.arr_0)
            y_val = torch.from_numpy(y_val.f.arr_0)

            x_combined = torch.cat([x_train, x_val], dim=0)
            y_combined = torch.cat([y_train, y_val], dim=0)

            train_data = torch.utils.data.TensorDataset(x_combined, y_combined)

        train_queue = DataLoader(
            train_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=2,
        )
        return train_queue

    def build_validation_data_loader(self) -> DataLoader:

        if self.hparams.task == 'pde':
            ntrain = 1000
            ntest = 100
            s = self.s
            r = self.hparams["sub"]

            TEST_PATH = os.path.join(self.download_directory, 'piececonst_r421_N1024_smooth2.mat')
            reader = MatReader(TEST_PATH)
            x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]
            y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]

            x_test = self.x_normalizer.encode(x_test)
            x_test = torch.cat([x_test.reshape(ntest, s, s, 1), self.grid.repeat(ntest, 1, 1, 1)], dim=3)

        elif self.hparams.task == 'protein':
            x_test = np.load('X_test.npz')
            y_test = np.load('Y_test.npz')
            x_test = torch.from_numpy(x_test.f.arr_0)
            y_test = torch.from_numpy(y_test.f.arr_0)

            f = open('psicov.json', )
            psicov = json.load(f)
            self.my_list = psicov['my_list']
            self.length_dict = psicov['length_dict']

        return DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                          batch_size=self.context.get_per_slot_batch_size(), shuffle=False, num_workers=2,)

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int
                    ) -> Dict[str, torch.Tensor]:


        x_train, y_train = batch
        self.model.train()
        logits = self.model(x_train)

        if self.hparams.task == 'pde':
            self.y_normalizer.cuda()
            target = self.y_normalizer.decode(y_train)
            logits = self.y_normalizer.decode(logits)
            loss = self.criterion(logits.view(logits.size(0), -1), target.view(logits.size(0), -1))
            mae = 0.0

        elif self.hparams.task == 'protein':
            loss = self.criterion(logits, y_train.squeeze())
            mae = F.l1_loss(logits, y_train.squeeze(), reduction='mean').item()


        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        return {
            'loss': loss,
            'MAE': mae,
        }


    def evaluate_full_dataset(
        self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:

        loss_sum = 0
        error_sum = 0
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                num_batches += 1
                logits = self.model(input)
                if self.hparams.task == 'pde':
                    self.y_normalizer.cuda()
                    logits = self.y_normalizer.decode(logits)
                    loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1)).item()
                    loss = loss / logits.size(0)

                elif self.hparams.task == 'protein':
                    print(target.shape)
                    print(logits.shape)

                    target = target.squeeze()
                    logits = logits.squeeze()
                    loss = self.criterion(logits, target)

                    mae = F.l1_loss(logits, logits, reduction='mean').item()

                    #target, logits, num = filter_MAE(target, logits, 8.0)
                    #error = self.error(logits, target)
                    #error = error / num
                    error_sum += mae

                loss_sum += loss

        results = {
            "validation_loss": loss_sum / num_batches,
            "MAE": error_sum / num_batches,
        }

        return results

