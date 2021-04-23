"""
This model is from the CNN NAS search space considered in:
    https://openreview.net/forum?id=S1eYHoC5FX

We will use the adaptive searcher in Determined to find a
good architecture in this search space for CIFAR-10.  
"""
import tempfile
from collections import namedtuple
from typing import Any, Dict, Sequence, Tuple, Union, cast
#from attrdict import AttrDict

import torch
import boto3
import os

from determined.pytorch import (
    DataLoader,
    LRScheduler,
    PyTorchTrial,
    PyTorchTrialContext,
)

import determined as det

from model import NetworkPoint as Network
import utils

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

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
        self.hparams = AttrDict(context.get_hparams())
        self.criterion = torch.nn.functional.cross_entropy
        # The last epoch is only used for logging.
        self._last_epoch = -1
        self.results = {"loss": float("inf"), "top1_accuracy": 0, "top5_accuracy": 0, "test_loss": float("inf"),
                        "test_top1_accuracy": 0, "test_top5_accuracy": 0}

        # Create a unique download directory for each rank so they don't overwrite each other.
        '''
        self.download_directory = tempfile.mkdtemp()

        if self.hparams['task'] == 'spherical':
            path = '/workspace/tasks/spherical/s2_mnist.gz'
            self.train_data, self.test_data = utils.load_spherical_data(path, self.context.get_per_slot_batch_size())

        if self.hparams['task'] == 'sEMG':
            self.download_directory = '/workspace/tasks/MyoArmbandDataset/PyTorchImplementation/sEMG'
        '''
        self.download_directory = self.download_data_from_s3()

        n_classes = 7 if self.hparams['task'] == 'sEMG' else 10
        in_channels = 3 if self.hparams['task'] == 'cifar' else 1

        # Define the model
        genotype = self.get_genotype_from_hps()
        self.model = self.context.wrap_model(
            Network(
                self.hparams["init_channels"],
                n_classes,
                self.hparams["layers"],
                self.hparams["auxiliary"],
                genotype,
                in_channels,
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

            self.train_data, self.val_data, self.test_data = utils.load_spherical_data(download_directory)

        if self.hparams.task == 'sEMG':
            data_files = ["saved_evaluation_dataset_test0.npy", "saved_evaluation_dataset_test1.npy",
                          "saved_evaluation_dataset_training.npy", "saved_pre_training_dataset_spectrogram.npy"]
            for data_file in data_files:
                filepath = os.path.join(download_directory, data_file)
                if not os.path.exists(filepath):
                    s3.download_file(s3_bucket, data_file, filepath)

        #instantiate test loader
        self.build_test_data_loader(download_directory)

        return download_directory

    def build_training_data_loader(self) -> DataLoader:
        if self.hparams['task'] == 'cifar':
            trainset, _ = utils.load_cifar_train_data(self.download_directory, self.hparams['permute'])

        elif self.hparams['task'] == 'spherical':
            trainset = self.train_data

        elif self.hparams['task'] == 'sEMG':
            trainset = utils.load_sEMG_train_data(self.download_directory)

        else:
            pass

        return DataLoader(trainset, batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self) -> DataLoader:


        if self.hparams['task'] == 'cifar':
            _, valset = utils.load_cifar_train_data(self.download_directory, self.hparams['permute'])

        elif self.hparams['task'] == 'spherical':
            valset = self.val_data

        elif self.hparams['task'] == 'sEMG':
            valset = utils.load_sEMG_val_data(self.download_directory)

        else:
            pass

        return DataLoader(valset, batch_size=self.context.get_per_slot_batch_size())

    def build_test_data_loader(self, download_directory):

        if self.hparams['task'] == 'cifar':
            testset = utils.load_cifar_test_data(download_directory, self.hparams['permute'])

        elif self.hparams['task'] == 'spherical':
            testset = self.test_data

        elif self.hparams['task'] == 'sEMG':
            testset = utils.load_sEMG_test_data(download_directory)

        else:
            pass

        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=self.context.get_per_slot_batch_size(),
                                                       shuffle=False, num_workers=2)
        return

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
        logits, logits_aux = self.model(input)
        loss = self.criterion(logits, target)
        if self.context.get_hparam("auxiliary"):
            loss_aux = self.criterion(logits_aux, target)
            loss += self.context.get_hparam("auxiliary_weight") * loss_aux
        top1, top5 = utils.accuracy(logits, target, topk=(1, 5))

        # Backward pass
        self.context.backward(loss)
        self.context.step_optimizer(
            optimizer=self.optimizer,
            clip_grads=lambda params: torch.nn.utils.clip_grad_norm_(
                params,
                self.context.get_hparam("clip_gradients_l2_norm"),
            ),
        )

        return {"loss": loss, "top1_accuracy": top1, "top5_accuracy": top5}

    def evaluate_full_dataset(
        self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        acc_top1 = 0
        acc_top5 = 0
        loss_avg = 0
        num_batches = 0
        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                num_batches += 1
                logits, _ = self.model(input)
                loss = self.criterion(logits, target)
                top1, top5 = utils.accuracy(logits, target, topk=(1, 5))
                acc_top1 += top1
                acc_top5 += top5
                loss_avg += loss
        results = {
            "loss": loss_avg.item() / num_batches,
            "top1_accuracy": acc_top1.item() / num_batches,
            "top5_accuracy": acc_top5.item() / num_batches,
        }

        test_acc_top1 = 0
        test_acc_top5 = 0
        test_loss_avg = 0
        num_batches = 0
        with torch.no_grad():
            for batch in self.test_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                num_batches += 1
                logits, _ = self.model(input)
                loss = self.criterion(logits, target)
                top1, top5 = utils.accuracy(logits, target, topk=(1, 5))
                test_acc_top1 += top1
                test_acc_top5 += top5
                test_loss_avg += loss

        results2 = {
            "test_loss": test_loss_avg.item() / num_batches,
            "test_top1_accuracy": test_acc_top1.item() / num_batches,
            "test_top5_accuracy": test_acc_top5.item() / num_batches,
        }

        results.update(results2)

        if results["top1_accuracy"] > self.results["top1_accuracy"]:
            self.results = results

        return self.results
