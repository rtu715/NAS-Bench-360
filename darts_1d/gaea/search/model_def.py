from typing import Any, Dict, Union, Sequence
import boto3
import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
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

from data import BilevelDataset
from model_search import Network
from optimizer import EG
from utils import AttrDict, accuracy, AverageMeter, calculate_stats

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
        self.hparams = AttrDict(trial_context.get_hparams())
        self.last_epoch = 0

        self.download_directory = self.download_data_from_s3()

        dataset_hypers = {'ECG': (4, 1), 'satellite': (24, 1), 'deepsea': (36, 4)}
        n_classes, in_channels = dataset_hypers[self.hparams.task]


        if self.hparams.task == 'deepsea':
            criterion = nn.BCEWithLogitsLoss().cuda()
            self.accuracy = False

        else: 
            criterion = nn.CrossEntropyLoss().cuda()
            self.accuracy = True

        # Initialize the models.
        self.model = self.context.wrap_model(
            Network(
                self.hparams.init_channels,
                n_classes,
                self.hparams.layers,
                criterion,
                self.hparams.nodes,
                k=self.hparams.shuffle_factor,
                in_channels=in_channels,
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
        #download_directory = '.'
        self.train_data, self.val_data, _ = load_data(self.hparams.task, download_directory, True)

        return download_directory

    def build_training_data_loader(self) -> DataLoader:
        """
        For bi-level NAS, we'll need each instance from the dataloader to have one image
        for training shared-weights and another for updating architecture parameters.
        """
        bilevel = BilevelDataset(self.train_data)
        self.train_data = bilevel
        print('Length of bilevel dataset: ', len(bilevel))

        return DataLoader(bilevel, batch_size=self.context.get_per_slot_batch_size(), shuffle=True, num_workers=2)

    def build_validation_data_loader(self) -> DataLoader:
        valset = self.val_data
        return DataLoader(valset, batch_size=self.context.get_per_slot_batch_size(), shuffle=False, num_workers=2)

    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        if epoch_idx != self.last_epoch:
            self.train_data.shuffle_val_inds()
        self.last_epoch = epoch_idx
        x_train, y_train, x_val, y_val = batch

        if self.hparams.task == 'deepsea':
            y_train = y_train.float()
            y_val = y_val.float()

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

    def evaluate_full_dataset(
            self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        if self.hparams.task == 'ECG':
            return self.evaluate_full_dataset_ECG(data_loader)

        elif self.hparams.task == 'satellite':
            return self.evaluate_full_dataset_satellite(data_loader)

        elif self.hparams.task == 'deepsea':
            return self.evaluate_full_dataset_deepsea(data_loader)

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
                loss = self.model._loss(input, target)
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
                loss = self.model._loss(input, target)
                top1, top5 = accuracy(logits, target, topk=(1,5))
                acc_top1.update(top1.item(), n)
                acc_top5.update(top5.item(), n)
                loss_avg.update(loss, n)

        results = {
            "loss": loss_avg.avg,
            "top1_accuracy": acc_top1.avg,
            "top5_accuracy": acc_top5.avg,
            }
        
        return results

    def evaluate_full_dataset_deepsea(
            self, data_loader: torch.utils.data.DataLoader
        ) -> Dict[str, Any]:
        
        loss_avg = AverageMeter()
        test_predictions = []
        test_gts = [] 
        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                n = input.size(0)
                logits = self.model(input)
                loss = self.model._loss(input, target.float())
                loss_avg.update(loss, n)
                logits_sigmoid = torch.sigmoid(logits)
                test_predictions.append(logits_sigmoid.detach().cpu().numpy())
                test_gts.append(target.detach().cpu().numpy()) 

        test_predictions = np.concatenate(test_predictions).astype(np.float32)
        test_gts = np.concatenate(test_gts).astype(np.int32)

        stats = calculate_stats(test_predictions, test_gts)
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])

        results = {
            "test_mAUC": mAUC,
            "test_mAP": mAP,
        }

        return results

    def build_callbacks(self):
        return {"genotype": GenotypeCallback(self.context)}
