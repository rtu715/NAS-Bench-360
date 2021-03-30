"""
This model is from the CNN NAS search space considered in:
    https://openreview.net/forum?id=S1eYHoC5FX

We will use the adaptive searcher in Determined to find a
good architecture in this search space for CIFAR-10.  
"""

from collections import namedtuple
from typing import Any, Dict
#from attrdict import AttrDict
import boto3
import os
import numpy as np


import torch

from determined.pytorch import (
    DataLoader,
    LRScheduler,
    PyTorchTrial,
    PyTorchTrialContext,
)

import determined as det

from model import NetworkPDE as Network
from utils import LpLoss, MatReader, UnitGaussianNormalizer

import utils

torch.backends.cudnn.enabled = False

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def apply_constraints(hparams, num_params):
    normal_skip_count = 0
    reduce_skip_count = 0
    normal_conv_count = 0
    for hp, val in hparams.items():
        if val == "skip_connect":
            if "normal" in hp:
                normal_skip_count += 1
            elif "reduce" in hp:
                reduce_skip_count += 1
        if val == "sep_conv_3x3":
            if "normal" in hp:
                normal_conv_count += 1

    # Reject if num skip_connect >= 3 or <1 in either normal or reduce cell.
    if normal_skip_count >= 3 or reduce_skip_count >= 3:
        raise det.InvalidHP("too many skip_connect operations")
    if normal_skip_count == 0 or reduce_skip_count == 0:
        raise det.InvalidHP("too few skip_connect operations")
    # Reject if fewer than 3 sep_conv_3x3 in normal cell.
    if normal_conv_count < 3:
        raise det.InvalidHP("fewer than 3 sep_conv_3x3 operations in normal cell")
    # Reject if num_params > 4.5 million or < 2.5 million.
    if num_params < 2.5e6 or num_params > 4.5e6:
        raise det.InvalidHP(
            "number of parameters in architecture is not between 2.5 and 4.5 million"
        )



class DARTSCNNTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.data_config = context.get_data_config()
        self.hparams = context.get_hparams()
        self.criterion = LpLoss(size_average=False)
        # The last epoch is only used for logging.
        self._last_epoch = -1
        self.results = {"loss": float("inf"), "top1_accuracy": 0, "top5_accuracy": 0}

        self.download_directory = self.download_data_from_s3()
        self.grid, self.s = utils.create_grid(self.hparams["sub"])



        # Define the model
        genotype = self.get_genotype_from_hps()
        self.model = self.context.wrap_model(
            Network(
                self.hparams["init_channels"],
                1,  # num_classes
                self.hparams["layers"],
                self.hparams["width"],
                self.hparams["auxiliary"],
                genotype,
            )
        )
        print("param size = {} MB".format(utils.count_parameters_in_MB(self.model)))
        size = 0
        for p in self.model.parameters():
            size += p.nelement()
        print("param count: {}".format(size))

        # Apply constraints if desired
        if "use_constraints" in self.hparams and self.hparams["use_constraints"]:
            apply_constraints(self.hparams, size)

        # Define the optimizer
        self.optimizer = self.context.wrap_optimizer(
            torch.optim.SGD(
                self.model.parameters(),
                lr=self.context.get_hparam("learning_rate"),
                momentum=self.context.get_hparam("momentum"),
                weight_decay=self.context.get_hparam("weight_decay"),
            )
        )

        # Define the LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            self.context.get_hparam("train_epochs"),
        )
        step_mode = LRScheduler.StepMode.STEP_EVERY_EPOCH
        self.wrapped_scheduler = self.context.wrap_lr_scheduler(
            self.scheduler, step_mode=step_mode
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

    def build_training_data_loader(self) -> Any:
        '''Load Darcy Flow data and normalize, preprocess'''

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

        return DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                          batch_size=self.context.get_per_slot_batch_size(), shuffle=True)

    def build_validation_data_loader(self) -> Any:
        '''Load Darcy Flow data for validation'''

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
                          batch_size=self.context.get_per_slot_batch_size(), shuffle=False)

    def get_genotype_from_hps(self):
        # This function creates an architecture definition
        # from the hyperparameter settings.
        cell_config = {"normal": [], "reduce": []}

        for cell in ["normal", "reduce"]:
            for node in range(4):
                for edge in [1, 2]:
                    edge_ind = self.hparams[
                        "{}_node{}_edge{}".format(cell, node + 1, edge)
                    ]
                    edge_op = self.hparams[
                        "{}_node{}_edge{}_op".format(cell, node + 1, edge)
                    ]
                    cell_config[cell].append((edge_op, edge_ind))
        print(cell_config)
        return Genotype(
            normal=cell_config["normal"],
            normal_concat=range(2, 6),
            reduce=cell_config["reduce"],
            reduce_concat=range(2, 6),
        )

    def train_batch(
        self, batch: Any, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        input, target = batch
        batch_size = self.context.get_per_slot_batch_size()

        self.model.drop_path_prob = (
            self.hparams["drop_path_prob"]
            * (self.scheduler.last_epoch)
            / self.hparams["train_epochs"]
        )
        if batch_idx == 0 or epoch_idx > self._last_epoch:
            print("epoch {} lr: {}".format(epoch_idx, self.scheduler.get_last_lr()[0]))
            print("drop_path_prob: {}".format(self.model.drop_path_prob))
        self._last_epoch = epoch_idx

        # Forward pass
        self.y_normalizer.cuda()
        logits, logits_aux = self.model(input)
        target = self.y_normalizer.decode(target)
        logits = self.y_normalizer.decode(logits)
        
        loss = self.criterion(logits.view(batch_size, -1), target.view(batch_size, -1))
        if self.context.get_hparam("auxiliary"):
            loss_aux = self.criterion(logits_aux, target)
            loss += self.context.get_hparam("auxiliary_weight") * loss_aux

        # Backward pass
        self.context.backward(loss)
        self.context.step_optimizer(
            optimizer=self.optimizer,
            clip_grads=lambda params: torch.nn.utils.clip_grad_norm_(
                params,
                self.context.get_hparam("clip_gradients_l2_norm"),
            ),
        )

        return {"loss": loss, }

    def evaluate_full_dataset(
        self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:

        loss_avg = 0
        num_batches = 0
        ntest = 100
        batch_size = self.context.get_per_slot_batch_size()

        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                num_batches += 1
                logits, _ = self.model(input)
                logits = self.y_normalizer.decode(logits)
                
                loss = self.criterion(logits.view(batch_size, -1), target.view(batch_size, -1)).item()
                loss_avg += loss

        results = {
            "validation_error": loss_avg / ntest,
        }

        return results
