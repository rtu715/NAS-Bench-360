from collections import namedtuple
from typing import Any, Dict

import os
import boto3

import torchvision.transforms as transforms
import numpy as np
from torch import nn

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
    AverageMeter,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

from data_utils.load_data import load_data
from data_utils.download_data import download_from_s3
from searched_genotypes import genotypes

Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class GAEAEvalTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.hparams = AttrDict(context.get_hparams())
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
        genotype = genotypes[self.context.get_hparam('task')]

        print(self.context.get_hparam('task'))
        print(genotype)

        dataset_hypers = {'ECG': (4, 1), 'satellite': (24, 1)}
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

    def get_genotype_from_hps(self):
        # only used in eval random archs
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

    def download_data_from_s3(self):
        '''Download data from s3 to store in temp directory'''

        s3_bucket = self.context.get_data_config()["bucket"]
        download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        s3 = boto3.client("s3")
        os.makedirs(download_directory, exist_ok=True)

        download_from_s3(s3_bucket, self.context.get_hparam('task'), download_directory)

        self.train_data, _, self.val_data = load_data(self.context.get_hparam('task'), download_directory, False, permute=self.context.get_hparam('permute'))

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
            #print("Epoch: {} lr {}".format(epoch_idx, current_lr))
       
            self.model.drop_path_prob = self.context.get_hparam("drop_path_prob") * epoch_idx / 600.0
            #print('current drop prob is {}'.format(self.model.drop_path_prob))
        
        self.last_epoch_idx = epoch_idx

        
        input, target = batch

        logits = self.model(input)
        loss = self.criterion(logits, target)

        self.context.backward(loss)
        self.context.step_optimizer(
            self.optimizer,
            clip_grads=lambda params: torch.nn.utils.clip_grad_norm_(
                params, self.context.get_hparam("clip_gradients_l2_norm"),
            ),
        )

        return {"loss": loss, }

    def evaluate_full_dataset(
            self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        if self.hparams.task == 'ECG':
            return self.evaluate_full_dataset_ECG(data_loader)

        elif self.hparams.task == 'satellite':
            return self.evaluate_full_dataset_satellite(data_loader)

        return None

    def evaluate_full_dataset_ECG(
            self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:

        loss_avg = AverageMeter()
        all_pred_prob = []
        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                n = input.size(0)
                logits = self.model(input)
                loss = self.criterion(logits, target)
                loss_avg.update(loss, n)
                all_pred_prob.append(logits.cpu().data.numpy())

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

        results = {
            "loss": loss_avg.avg,
            "score": f1_score,
        }

        return results

    def evaluate_full_dataset_satellite(
            self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:

        acc_top1 = AverageMeter()
        acc_top5 = AverageMeter()
        loss_avg = AverageMeter()
        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                n = input.size(0)
                logits = self.model(input)
                loss = self.criterion(logits, target)
                top1, top5 = accuracy(logits, target, topk=(1, 5))
                acc_top1.update(top1.item(), n)
                acc_top5.update(top5.item(), n)
                loss_avg.update(loss, n)

        results = {
            "loss": loss_avg.avg,
            "top1_accuracy": acc_top1.avg,
            "top5_accuracy": acc_top5.avg,
        }


        return results

