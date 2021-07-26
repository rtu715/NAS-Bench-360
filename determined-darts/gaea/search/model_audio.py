from typing import Any, Dict, Union, Sequence
import boto3
import os
import tempfile
import numpy as np
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_utils.audio_dataset import *
from data_utils.audio_dataset import _collate_fn, _collate_fn_eval


from determined.pytorch import (
    PyTorchTrial,
    PyTorchTrialContext,
    DataLoader,
    LRScheduler,
    PyTorchCallback
)

from data import BilevelAudioDataset
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
        #self.data_config = trial_context.get_data_config()
        self.hparams = utils.AttrDict(trial_context.get_hparams())
        self.last_epoch = 0

        '''
        self.download_directory = tempfile.mkdtemp()

        if self.hparams.task == 'spherical':
            path = '/workspace/tasks/spherical/s2_mnist.gz'
            self.train_data, self.test_data = utils.load_spherical_data(path, self.context.get_per_slot_batch_size())

        if self.hparams.task == 'sEMG':
            self.download_directory = '/workspace/tasks/MyoArmbandDataset/PyTorchImplementation/sEMG'
        '''
        self.download_directory = self.download_data_from_s3()

        dataset_hypers = {'audio': (200, 1)}
        n_classes, in_channels = dataset_hypers[self.hparams.task]


        # Initialize the models.
        self.criterion = nn.BCEWithLogitsLoss().cuda()
        self.model = self.context.wrap_model(
            Network(
                self.hparams.init_channels,
                n_classes,
                self.hparams.layers,
                self.criterion,
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
        bilevel = BilevelAudioDataset(trainset)
        self.train_data = bilevel
        print('Length of bilevel dataset: ', len(bilevel))

        return DataLoader(bilevel, batch_size=self.context.get_per_slot_batch_size(), shuffle=True, sampler=None,
                          collate_fn=_collate_fn,
                          pin_memory=False, drop_last=True, num_workers=4)

    def build_validation_data_loader(self) -> DataLoader:

        valset = self.val_data

        return DataLoader(valset, sampler=None, num_workers=4,
                          collate_fn=_collate_fn_eval,
                          shuffle=False, batch_size=1,
                          pin_memory=False)


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

        loss_avg = utils.AverageMeter()
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
                loss_avg.update(loss, n)

                logits_sigmoid = torch.sigmoid(logits)
                val_predictions.append(logits_sigmoid.detach().cpu().numpy()[0])
                val_gts.append(target.detach().cpu().numpy()[0])

        val_preds = np.asarray(val_predictions).astype('float32')
        val_gts = np.asarray(val_gts).astype('int32')
        map_value = average_precision_score(val_gts, val_preds, average="macro")

        results = {
            "loss": loss_avg.avg,
            'val_mAP': map_value,
        }

        if self.last_epoch % 10 == 0:
            test_predictions = []
            test_gts = []
            for ix in range(self.testset.len):
                with torch.no_grad():
                    batch = self.testset[ix]
                    x, y = batch
                    x = x.cuda()
                    y_pred = self.model(x)
                    y_pred = y_pred.mean(0).unsqueeze(0)
                    sigmoid_preds = torch.sigmoid(y_pred)
                test_predictions.append(sigmoid_preds.detach().cpu().numpy()[0])
                test_gts.append(y.detach().cpu().numpy()[0])  # drop batch axis
            test_predictions = np.asarray(test_predictions).astype('float32')
            test_gts = np.asarray(test_gts).astype('int32')

            stats = calculate_stats(test_predictions, test_gts)
            mAP = np.mean([stat['AP'] for stat in stats])
            mAUC = np.mean([stat['auc'] for stat in stats])

            results2 = {
                "test_mAUC": mAP,
                'test_mAP': mAUC,
            }

            results.update(results2)

        return results

    def build_callbacks(self):
        return {"genotype": GenotypeCallback(self.context)}
