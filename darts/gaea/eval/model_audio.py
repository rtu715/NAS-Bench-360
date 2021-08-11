from collections import namedtuple
from typing import Any, Dict

import os
import boto3
import numpy as np

import torchvision.transforms as transforms
from torch import nn
from sklearn.metrics import average_precision_score

from determined.pytorch import (
    DataLoader,
    LRScheduler,
    PyTorchTrial,
    PyTorchTrialContext,
)

from model import Network

from data_utils.audio_dataset import *
from data_utils.audio_dataset import _collate_fn, _collate_fn_eval

from torch.optim.lr_scheduler import CosineAnnealingLR

import utils

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

        self.criterion = nn.BCEWithLogitsLoss().cuda()
        download_directory = self.download_data_from_s3()

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
                150.0,
                0,
            ),
            step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH,
        )

    def build_model_from_config(self):
        genotype = genotypes[self.context.get_hparam('task')]

        #genotype = self.get_genotype_from_hps()

        print(self.context.get_hparam('task'))
        print(genotype)

        dataset_hypers = {'audio': (200, 1)}
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
        #download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        s3 = boto3.client("s3")
        #os.makedirs(download_directory, exist_ok=True)

        download_directory = '.'
        download_from_s3(s3_bucket, self.context.get_hparam('task'), download_directory)
        self.train_data, _ , self.val_data = load_data(self.context.get_hparam('task'), download_directory,
                                                       False, permute=self.context.get_hparam('permute'))

        return download_directory



    def build_training_data_loader(self) -> DataLoader:

        train_data = self.train_data

        train_queue = DataLoader(train_data, num_workers=4, batch_size=self.context.get_per_slot_batch_size(),
                          shuffle=True, sampler=None, collate_fn=_collate_fn,
                          pin_memory=False, drop_last=True)
        return train_queue

    def build_validation_data_loader(self) -> DataLoader:

        valid_data = self.val_data

        valid_queue = DataLoader(valid_data, sampler=None, num_workers=4,
                          collate_fn=_collate_fn_eval,
                          shuffle=False, batch_size=1,
                          pin_memory=False
                          )
        return valid_queue

    def train_batch(
        self, batch: Any, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        if batch_idx == 0 or self.last_epoch_idx < epoch_idx:
            current_lr = self.lr_scheduler.get_last_lr()[0]
            print("Epoch: {} lr {}".format(epoch_idx, current_lr))
       
            self.model.drop_path_prob = self.context.get_hparam("drop_path_prob") * epoch_idx / 150.0
            print('current drop prob is {}'.format(self.model.drop_path_prob))
        
        self.last_epoch_idx = epoch_idx

        input, _, target = batch
        print(input.shape)
        logits = self.model(input)
        loss = self.criterion(logits, target)

        self.context.backward(loss)
        self.context.step_optimizer(
            self.optimizer,
            clip_grads=lambda params: torch.nn.utils.clip_grad_norm_(
                params, self.context.get_hparam("clip_gradients_l2_norm"),
            ),
        )

        return {"loss": loss,}


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

        stats = calculate_stats(val_preds, val_gts)
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])

        results = {
            "test_loss": loss_avg.avg,
            "test_mAUC": mAUC,
            "test_mAP": mAP,
        }

        return results

