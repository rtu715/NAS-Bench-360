from typing import Any, Dict, Union, Sequence
import os
from collections import namedtuple
import boto3
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from determined.pytorch import (
    PyTorchTrial,
    PyTorchTrialContext,
    DataLoader,
    LRScheduler,
    PyTorchCallback
)

from data import BilevelDataset
#from model_search import Network
from model_search_expansion import Network
from model_eval import DiscretizedNetwork
#from model_eval_expansion import DiscretizedNetwork
from optimizer import EG
from utils import AttrDict, LpLoss, MatReader, UnitGaussianNormalizer
from utils import LogCoshLoss
import utils

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]
Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")


class GenotypeCallback(PyTorchCallback):
    def __init__(self, context):
        self.model = context.models[0]
        self.search_phase = context.get_hparam('train')
    
    def on_validation_end(self, metrics):
        if self.search_phase:
            print(self.model.genotype())

        else:
            print('eval phase - constant genotype')


class GAEASearchTrial(PyTorchTrial):
    def __init__(self, trial_context: PyTorchTrialContext) -> None:
        self.context = trial_context
        self.data_config = trial_context.get_data_config()
        self.hparams = AttrDict(trial_context.get_hparams())
        self.last_epoch = 0

        self.download_directory = self.download_data_from_s3()

        if self.hparams.task == 'pde':
            self.grid, self.s = utils.create_grid(self.hparams["sub"])
            self.criterion = LpLoss(size_average=False)
            self.in_channels = 3
            self.n_classes = 1

        elif self.hparams.task == 'protein':
            self.criterion = LogCoshLoss()
            #error is reported via MAE
            self.error = nn.L1Loss(reduction='sum')
            self.in_channels = 57
            self.n_classes = 1

        else:
            raise NotImplementedError

        # Initialize the models.
        if self.hparams.train:
            self.model = self.context.wrap_model(
                Network(
                    self.hparams.init_channels,
                    self.n_classes,
                    self.hparams.layers,
                    self.criterion,
                    self.hparams.nodes,
                    self.hparams.multiplier,
                    self.in_channels,
                    k=self.hparams.shuffle_factor,
                )
            )

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

        else:
            if self.hparams.task == 'pde':
                #the genotype that works
                searched_genotype=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
                #from search without expansion
                #searched_genotype= Genotype(normal=[('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('dil_conv_3x3', 1), ('dil_conv_5x5', 4), ('dil_conv_3x3', 2)], normal_concat=range(5, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(5, 6))

                #searched with expansion
                #searched_genotype = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
            
            else:
                raise ValueError

            print(searched_genotype)

            model = DiscretizedNetwork(
                self.hparams.init_channels,
                self.n_classes,
                self.hparams.layers,
                searched_genotype,
                in_channels=self.in_channels,
                drop_path_prob=self.context.get_hparam("drop_path_prob"),
            )

            self.model = self.context.wrap_model(model)

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
                    self.hparams.scheduler_epochs,
                    self.hparams.min_learning_rate,
                ),
                step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH,
            )
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)/ 1e6
        print('Parameter size in MB: ', total_params)

    def download_data_from_s3(self):
        '''Download pde data from s3 to store in temp directory'''

        s3_bucket = self.context.get_data_config()["bucket"]
        download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        
        if self.hparams.task == 'pde':
            data_files = ["piececonst_r421_N1024_smooth1.mat", "piececonst_r421_N1024_smooth2.mat"]
            s3_path = None

        elif self.hparams.task == 'protein':
            data_files =['X_train.npz', 'X_valid.npz', 'Y_train.npz', 'Y_valid.npz']
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
        """
        For bi-level NAS, we'll need each instance from the dataloader to have one image
        for training shared-weights and another for updating architecture parameters.
        """

        if self.hparams.task =='pde':
            TRAIN_PATH = os.path.join(self.download_directory, 'piececonst_r421_N1024_smooth1.mat')
            self.reader = MatReader(TRAIN_PATH)
            s = self.s
            r = self.hparams["sub"]
            ntrain = 1000
            ntest = 100
            if self.hparams.train:
                x_train = self.reader.read_field('coeff')[:ntrain-ntest, ::r, ::r][:, :s, :s]
                y_train = self.reader.read_field('sol')[:ntrain-ntest, ::r, ::r][:, :s, :s]

                self.x_normalizer = UnitGaussianNormalizer(x_train)
                x_train = self.x_normalizer.encode(x_train)

                self.y_normalizer = UnitGaussianNormalizer(y_train)
                y_train = self.y_normalizer.encode(y_train)

                ntrain = ntrain - ntest
                x_train = torch.cat([x_train.reshape(ntrain, s, s, 1), self.grid.repeat(ntrain, 1, 1, 1)], dim=3)
                train_data = torch.utils.data.TensorDataset(x_train, y_train)

            else:
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
            
            if self.hparams.train:
                #save 100 samples from train set as validation
                x_train = x_train[100:]
                y_train = y_train[100:]
                
    
            train_data = torch.utils.data.TensorDataset(x_train, y_train)
        
        print(x_train.shape)
        print(y_train.shape)
        bilevel_data = BilevelDataset(train_data)

        self.train_data = bilevel_data if self.hparams.train else train_data

        train_queue = DataLoader(
            self.train_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=2,
        )
        return train_queue

    def build_validation_data_loader(self) -> DataLoader:
        
        if self.hparams.task == 'pde':
            ntrain= 1000
            ntest = 100
            s = self.s
            r = self.hparams["sub"]

            if self.hparams.train:
                x_test = self.reader.read_field('coeff')[ntrain-ntest:ntrain, ::r, ::r][:, :s, :s]
                y_test = self.reader.read_field('sol')[ntrain-ntest:ntrain, ::r, ::r][:, :s, :s]

                x_test = self.x_normalizer.encode(x_test)
                x_test = torch.cat([x_test.reshape(ntest, s, s, 1), self.grid.repeat(ntest, 1, 1, 1)], dim=3)

            else:
                TEST_PATH = os.path.join(self.download_directory, 'piececonst_r421_N1024_smooth1.mat')
                reader = MatReader(TEST_PATH)
                x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]
                y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]

                x_test = self.x_normalizer.encode(x_test)
                x_test = torch.cat([x_test.reshape(ntest, s, s, 1), self.grid.repeat(ntest, 1, 1, 1)], dim=3)

        elif self.hparams.task == 'protein': 
            
            if self.hparams.train: 
                x_train = np.load('X_train.npz')
                y_train = np.load('Y_train.npz')
                
                x_train = torch.from_numpy(x_train.f.arr_0) 
                y_train = torch.from_numpy(y_train.f.arr_0)
                x_test = x_train[:100]
                y_test = y_train[:100]

            else:
                x_test = np.load('X_valid.npz')
                y_test = np.load('Y_valid.npz')
                x_test = torch.from_numpy(x_test.f.arr_0)
                y_test = torch.from_numpy(y_test.f.arr_0)

            print(x_test.shape)
        return DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                          batch_size=self.context.get_per_slot_batch_size(), shuffle=False, num_workers=2,)



    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        if self.hparams.train:
            if epoch_idx != self.last_epoch:
                self.train_data.shuffle_val_inds()
            self.last_epoch = epoch_idx
            x_train, y_train, x_val, y_val = batch
        
        else:
            x_train, y_train = batch

        batch_size = self.context.get_per_slot_batch_size()

        if self.hparams.train:
            # Train shared-weights
            for a in self.model.arch_parameters():
                a.requires_grad = False
            for w in self.model.ws_parameters():
                w.requires_grad = True

            logits = self.model(x_train)
            
            if self.hparams.task =='pde':
                self.y_normalizer.cuda()
                target = self.y_normalizer.decode(y_train)
                logits = self.y_normalizer.decode(logits)
                loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1))
            
            elif self.hparams.task == 'protein':
                #logits = torch.clamp(logits, min=0.01)        
                loss = self.criterion(logits, y_train.squeeze())

            else:
                raise NotImplementedError

            self.context.backward(loss)

            self.context.step_optimizer(
                optimizer=self.ws_opt,
                clip_grads=lambda params: torch.nn.utils.clip_grad_norm_(
                    params,
                    self.context.get_hparam("clip_gradients_l2_norm"),
                ),
            )

            arch_loss = 0.0
            if epoch_idx > 10:
                # Train arch parameters
                for a in self.model.arch_parameters():
                    a.requires_grad = True
                for w in self.model.ws_parameters():
                    w.requires_grad = False

                logits = self.model(x_val)
                if self.hparams.task =='pde':
                    target = self.y_normalizer.decode(y_val)
                    logits = self.y_normalizer.decode(logits)
                    arch_loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1))
                elif self.hparams.task =='protein': 
                    arch_loss = self.criterion(logits, y_val.squeeze())

                self.context.backward(arch_loss)
                self.context.step_optimizer(self.arch_opt)

        else: 
            if self.hparams.task =='pde':
                self.y_normalizer.cuda()
                logits = self.model(x_train)
                target = self.y_normalizer.decode(y_train)
                logits = self.y_normalizer.decode(logits)
                loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1))
            elif self.hparams.task =='protein':
                logits = self.model(x_train)
                loss = self.criterion(logits, y_train.squeeze())
            
            self.context.backward(loss)
            #self.context.step_optimizer(self.optimizer)
            self.context.step_optimizer(
                optimizer=self.optimizer,
                clip_grads=lambda params: torch.nn.utils.clip_grad_norm_(
                    params,
                    self.context.get_hparam("clip_gradients_l2_norm"),
                ),
            )
            arch_loss = 0.0

        return {
            "loss": loss,
            "arch_loss": arch_loss,
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
                    logits = self.y_normalizer.decode(logits)
                    loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1)).item()
                    loss = loss / logits.size(0)
                    error = 0 

                elif self.hparams.task == 'protein':
                    target = target.squeeze()
                    loss = self.criterion(logits, target)
                    loss = loss / logits.size(0)
                    
                    #filter the matrixes
                    target, logits, num = utils.filter_MAE(target, logits, 8.0)
                    error = self.error(logits, target)
                    error = error / num

                loss_sum += loss
                error_sum += error

        results = {
            "validation_error": loss_sum / num_batches,
            "MAE": error_sum / num_batches,
        }

        return results

    def build_callbacks(self):
        return {"genotype": GenotypeCallback(self.context)}
