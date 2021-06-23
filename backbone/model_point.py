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

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, LRScheduler
from backbone_pt import Backbone_Pt

import utils_pt

from data_utils.load_data import load_data
from data_utils.download_data import download_from_s3

# Constants about the dataset here (need to modify)

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


def accuracy_rate(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the accuracy rate based on dense predictions and sparse labels."""
    assert len(predictions) == len(labels), "Predictions and labels must have the same length."
    assert len(labels.shape) == 1, "Labels must be a column vector."

    return (  # type: ignore
            float((predictions.argmax(1) == labels.to(torch.long)).sum()) / predictions.shape[0]
    )


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
        #self.results = {"loss": float("inf"), "top1_accuracy": 0, "top5_accuracy": 0, "test_loss": float("inf"),
        #                "test_top1_accuracy": 0, "test_top5_accuracy": 0}

        self.criterion = nn.CrossEntropyLoss().cuda()


        dataset_hypers = {'sEMG': (7, 1), 'ninapro': (18, 1), 'cifar10': (10, 3), 'smnist': (10, 1), 'cifar100':(100, 3), 'scifar100': (100, 3)}

        n_classes, in_channels = dataset_hypers[self.hparams.task]
        print('task: ', self.hparams.task, 'in_channels: ',  in_channels, 'classes: ', n_classes)
        # Changing our backbone
        depth = list(map(int, self.hparams.backbone.split(',')))[0]
        width = list(map(int, self.hparams.backbone.split(',')))[1]

        self.backbone = Backbone_Pt(
            depth,
            n_classes,
            width,
            dropRate=self.hparams.droprate,
            in_channels=in_channels,
        )

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
            self.train_data, self.val_data, self.test_data = load_data(self.hparams.task, download_directory, True, self.hparams.permute) 
            self.build_test_data_loader(download_directory)
        
        else:
            self.train_data, _, self.val_data = load_data(self.hparams.task, download_directory, False, self.hparams.permute)

        return download_directory

    def build_training_data_loader(self) -> DataLoader:

        trainset = self.train_data
        print(len(trainset))
        return DataLoader(trainset, batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self) -> DataLoader:

        valset = self.val_data
        print(len(valset))

        return DataLoader(valset, batch_size=self.context.get_per_slot_batch_size())

    def build_test_data_loader(self, download_directory):

        testset = self.test_data
        print(len(testset))
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=self.context.get_per_slot_batch_size(),
                                                       shuffle=False, num_workers=2)
        return
    
    '''
    Train and Evaluate Methods
    '''

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int
                    ) -> Dict[str, torch.Tensor]:

        x_train, y_train = batch
        self.model.train()
        output = self.model(x_train)
        loss = self.criterion(output, y_train)
        top1, top5 = utils_pt.accuracy(output, y_train, topk=(1, 5))


        self.context.backward(loss)
        self.context.step_optimizer(self.opt)

        return {
            'loss': loss,
            'top1_accuracy': top1.item(),
            'top5_accuracy': top5.item(),
        }
    
    def evaluate_full_dataset(
        self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        acc_top1 = utils_pt.AverageMeter()
        acc_top5 = utils_pt.AverageMeter()
        loss_avg = utils_pt.AverageMeter()
        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                n = input.size(0)
                logits = self.model(input)
                loss = self.criterion(logits, target)
                top1, top5 = utils_pt.accuracy(logits, target, topk=(1, 5))
                acc_top1.update(top1.item(), n)
                acc_top5.update(top5.item(), n)
                loss_avg.update(loss, n)
        results = {
            "loss": loss_avg.avg,
            "top1_accuracy": acc_top1.avg,
            "top5_accuracy": acc_top5.avg,
        }
        
        if self.hparams.train:
            test_acc_top1 = utils_pt.AverageMeter()
            test_acc_top5 = utils_pt.AverageMeter()
            test_loss = utils_pt.AverageMeter()
            with torch.no_grad():
                for batch in self.test_loader:
                    batch = self.context.to_device(batch)
                    input, target = batch
                    n = input.size(0)
                    logits = self.model(input)
                    loss = self.criterion(logits, target)
                    top1, top5 = utils_pt.accuracy(logits, target, topk=(1, 5))
                    test_acc_top1.update(top1.item(), n)
                    test_acc_top5.update(top5.item(), n)
                    test_loss.update(loss, n)

            results2 = {
                "test_loss": test_loss.avg,
                "test_top1_accuracy": test_acc_top1.avg,
                "test_top5_accuracy": test_acc_top5.avg,
            }

            results.update(results2)


        return results
