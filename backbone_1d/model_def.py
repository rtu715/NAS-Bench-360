'''
Determined model def example:
https://github.com/determined-ai/determined/tree/master/examples/computer_vision/cifar10_pytorch
'''
import tempfile
from typing import Any, Dict, Sequence, Tuple, Union, cast
from functools import partial

import os
import boto3

import torch
from torch import nn
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, LRScheduler
from backbone_1d import ResNet1D

import utils

from data_utils.load_data import load_data
from data_utils.download_data import download_from_s3

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class BackboneTrial(PyTorchTrial):
    def __init__(self, trial_context: PyTorchTrialContext) -> None:
        self.context = trial_context
        self.hparams = AttrDict(trial_context.get_hparams())
        self.last_epoch = 0

        self.download_directory = self.download_data_from_s3()

        dataset_hypers = {'ECG': (4, 1), 'satellite': (24, 1), 'deepsea': (36, 4)}

        if self.hparams.task == 'deepsea':
            self.criterion = nn.BCEWithLogitsLoss().cuda()
            self.accuracy = False

        else: 
            self.criterion = nn.CrossEntropyLoss().cuda()
            self.accuracy = True

        n_classes, in_channels = dataset_hypers[self.hparams.task]
        print('task: ', self.hparams.task, 'in_channels: ',  in_channels, 'classes: ', n_classes)
        
        self.backbone = ResNet1D(in_channels, 64, n_classes)

        total_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)/ 1e6
        print('Parameter size in MB(backbone): ', total_params)

        self.model = self.context.wrap_model(self.backbone)

        '''
        Definition of optimizer 
        '''
        nesterov = self.hparams.nesterov if self.hparams.momentum else False

        self.opt = self.context.wrap_optimizer(torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            nesterov=nesterov)
            )

        self.lr_scheduler = self.context.wrap_lr_scheduler(
            lr_scheduler=torch.optim.lr_scheduler.LambdaLR(
                self.opt,
                lr_lambda=self.weight_sched,
                last_epoch=self.hparams.start_epoch - 1
            ),
            step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH,
        )


    def weight_sched(self, epoch) -> Any:
        if self.hparams.epochs != 200:
            return 0.2 ** (epoch >= int(0.3 * self.hparams.epochs)) * 0.2 ** (epoch > int(0.6 * self.hparams.epochs)) * 0.2 ** (epoch > int(0.8 * self.hparams.epochs))
        #print('using original weight schedule') 
        return 0.2 ** (epoch >= 60) * 0.2 ** (epoch >= 120) * 0.2 ** (epoch >=160)


    def download_data_from_s3(self):
        '''Download data from s3 to store in temp directory'''
        s3_bucket = self.context.get_data_config()["bucket"]
        download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        s3 = boto3.client("s3")
        os.makedirs(download_directory, exist_ok=True)

        download_from_s3(s3_bucket, self.hparams.task, download_directory)
        if self.hparams.train:
            self.train_data, self.val_data, self.test_data = load_data(self.hparams.task, download_directory, True) 
        
        else:
            self.train_data, _, self.val_data = load_data(self.hparams.task, download_directory, False)

        print('train size: %d, val size: %d' % (len(self.train_data), len(self.val_data)))
        return download_directory


    def build_training_data_loader(self) -> DataLoader:

        train_data = self.train_data
        train_queue = DataLoader(
            train_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )
        return train_queue

    def build_validation_data_loader(self) -> DataLoader:

        valid_data = self.val_data
        valid_queue = DataLoader(
            valid_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        return valid_queue

    '''
    Train and Evaluate Methods
    '''

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int
                    ) -> Dict[str, torch.Tensor]:

        x_train, y_train = batch
        self.model.train()
        output = self.model(x_train)

        if self.hparams.task == 'deepsea':
            y_train = y_train.float()

        loss = self.criterion(output, y_train)

        self.context.backward(loss)
        self.context.step_optimizer(self.opt)

        results = {
            'loss': loss,
        }
        
        if self.accuracy: 
            top1 = utils.accuracy(output, y_train, topk=(1,))[0]
            results['top1_accuracy'] = top1.item()
        
        return results

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

        loss_avg = utils.AverageMeter()
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

    def evaluate_full_dataset_deepsea(
            self, data_loader: torch.utils.data.DataLoader
        ) -> Dict[str, Any]:
        
        loss_avg = utils.AverageMeter()
        test_predictions = []
        test_gts = [] 
        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                n = input.size(0)
                logits = self.model(input)
                loss = self.criterion(logits, target.float())
                loss_avg.update(loss, n)
                logits_sigmoid = torch.sigmoid(logits)
                test_predictions.append(logits_sigmoid.detach().cpu().numpy())
                test_gts.append(target.detach().cpu().numpy()) 

        test_predictions = np.concatenate(test_predictions).astype(np.float32)
        test_gts = np.concatenate(test_gts).astype(np.int32)

        stats = utils.calculate_stats(test_predictions, test_gts)
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])

        results = {
            "test_mAUC": mAUC,
            "test_mAP": mAP,
        }

        return results
