import os
import pprint
import sys
import boto3
import json
from typing import Any, Dict, Sequence, Union
import tarfile

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tools.lr_scheduler import get_lr_scheduler

from determined.pytorch import (
    PyTorchTrial,
    PyTorchTrialContext,
    DataLoader,
    LRScheduler
)

from configs.grid_train_cfg import cfg as config
from generate_random import generate_arch
from models import model_derived
from tools import utils
from utils_grid import LpLoss, MatReader, UnitGaussianNormalizer, AverageMeter
from utils_grid import create_grid, calculate_mae, maskMetric, set_input
from data_utils.protein_io import load_list
from data_utils.protein_gen import PDNetDataset
from data_utils.cosmic_dataset import PairedDatasetImagePath, get_dirs

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class DenseNASSearchTrial(PyTorchTrial):

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
            #self.criterion = LogCoshLoss()
            self.criterion = nn.MSELoss(reduction='mean')

            # error is reported via MAE
            self.error = nn.L1Loss(reduction='sum')
            self.in_channels = 57

        elif self.hparams.task == 'cosmic':
            self.criterion = nn.BCEWithLogitsLoss()
            self.in_channels = 1

        else:
            raise NotImplementedError

        cudnn.benchmark = True
        cudnn.enabled = True

        config.net_config, config.net_type = self.hparams.net_config, self.hparams.net_type
        derivedNetwork = getattr(model_derived, '%s_Net' % self.hparams.net_type.upper())
        self.rand_arch = generate_arch(self.hparams.task, self.hparams.net_type, self.hparams.target_arch)
        model = derivedNetwork(self.rand_arch, task=self.hparams.task, config=config)
  
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

        scheduler = get_lr_scheduler(config, self.optimizer, self.hparams.num_examples, self.context.get_per_slot_batch_size())
        scheduler.last_step = 0

        self.scheduler = self.context.wrap_lr_scheduler(scheduler, step_mode=LRScheduler.StepMode.MANUAL_STEP)

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
            data_files = ['protein.zip']
            data_dir = download_directory
            self.all_feat_paths = [data_dir + '/deepcov/features/',
                              data_dir + '/psicov/features/', data_dir + '/cameo/features/']
            self.all_dist_paths = [data_dir + '/deepcov/distance/',
                              data_dir + '/psicov/distance/', data_dir + '/cameo/distance/']
            s3_path = None

        elif self.hparams.task == 'cosmic':
            data_files = ['deepCR.ACS-WFC.train.tar', 'deepCR.ACS-WFC.test.tar']
            s3_path = 'cosmic'

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
            x_train = self.reader.read_field('coeff')[:ntrain - ntest, ::r, ::r][:, :s, :s]
            y_train = self.reader.read_field('sol')[:ntrain - ntest, ::r, ::r][:, :s, :s]

            self.x_normalizer = UnitGaussianNormalizer(x_train)
            x_train = self.x_normalizer.encode(x_train)

            self.y_normalizer = UnitGaussianNormalizer(y_train)
            y_train = self.y_normalizer.encode(y_train)

            ntrain = ntrain - ntest
            x_train = torch.cat([x_train.reshape(ntrain, s, s, 1), self.grid.repeat(ntrain, 1, 1, 1)], dim=3)
            train_data = torch.utils.data.TensorDataset(x_train, y_train)

        elif self.hparams.task == 'protein':
            os.chdir(self.download_directory)
            import zipfile
            with zipfile.ZipFile('protein.zip', 'r') as zip_ref:
                zip_ref.extractall()

            self.deepcov_list = load_list('deepcov.lst', -1)

            self.length_dict = {}
            for pdb in self.deepcov_list:
                (ly, seqy, cb_map) = np.load(
                    'deepcov/distance/' + pdb + '-cb.npy',
                    allow_pickle=True)
                self.length_dict[pdb] = ly

            train_pdbs = self.deepcov_list[100:]

            train_data = PDNetDataset(train_pdbs, self.all_feat_paths, self.all_dist_paths,
                                      128, 10, self.context.get_per_slot_batch_size(), 57,
                                      label_engineering = '16.0')

        elif self.hparams.task == 'cosmic':
            # extract tar file and get directories
            # base_dir = '/workspace/tasks/cosmic/deepCR.ACS-WFC'
            base_dir = self.download_directory
            print(base_dir)

            os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
            data_base = os.path.join(base_dir, 'data')
            train_tar = tarfile.open(os.path.join(base_dir, 'deepCR.ACS-WFC.train.tar'))
            test_tar = tarfile.open(os.path.join(base_dir, 'deepCR.ACS-WFC.test.tar'))

            train_tar.extractall(data_base)
            test_tar.extractall(data_base)
            get_dirs(base_dir, data_base)

            self.train_dirs = np.load(os.path.join(base_dir, 'train_dirs.npy'), allow_pickle=True)
            self.test_dirs = np.load(os.path.join(base_dir, 'test_dirs.npy'), allow_pickle=True)

            aug_sky = (-0.9, 3)

            # only train f435 and GAL flag for now
            print(self.train_dirs[0])
            train_data = PairedDatasetImagePath(self.train_dirs[::], aug_sky[0], aug_sky[1], part='train')

            self.data_shape = train_data[0][0].shape[1]
            print(len(train_data))

        train_queue = DataLoader(
            train_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=2,
        )
        return train_queue

    def build_validation_data_loader(self) -> DataLoader:
        batch_size = self.context.get_per_slot_batch_size()

        if self.hparams.task == 'pde':
            ntrain = 1000
            ntest = 100
            s = self.s
            r = self.hparams["sub"]

            x_test = self.reader.read_field('coeff')[ntrain - ntest:ntrain, ::r, ::r][:, :s, :s]
            y_test = self.reader.read_field('sol')[ntrain - ntest:ntrain, ::r, ::r][:, :s, :s]

            x_test = self.x_normalizer.encode(x_test)
            x_test = torch.cat([x_test.reshape(ntest, s, s, 1), self.grid.repeat(ntest, 1, 1, 1)], dim=3)

            valid_queue = DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                     batch_size=self.context.get_per_slot_batch_size(), shuffle=False, num_workers=2, )


        elif self.hparams.task == 'protein':
            valid_pdbs = self.deepcov_list[:100]
            valid_data = PDNetDataset(valid_pdbs, self.all_feat_paths, self.all_dist_paths,
                                      128, 10, self.context.get_per_slot_batch_size(), 57,
                                      label_engineering='16.0')
            valid_queue = DataLoader(valid_data, batch_size=2, shuffle=True, num_workers=2)

        elif self.hparams.task == 'cosmic':
            aug_sky = (-0.9,3)
            valid_data = PairedDatasetImagePath(self.train_dirs[::], aug_sky[0], aug_sky[1], part='test')
            print(len(valid_data))
            valid_queue = DataLoader(valid_data, batch_size=self.context.get_per_slot_batch_size(), shuffle=False, num_workers=8)

        else:
            raise NotImplementedError

        return valid_queue

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int
                    ) -> Dict[str, torch.Tensor]:

        self.scheduler.step()
        self.model.train()

        if self.hparams.task == 'pde':
            x_train, y_train = batch
            logits = self.model(x_train)
            self.y_normalizer.cuda()
            target = self.y_normalizer.decode(y_train.squeeze())
            logits = self.y_normalizer.decode(logits.squeeze())
            loss = self.criterion(logits.view(logits.size(0), -1), target.view(logits.size(0), -1))
            mae = 0.0

        elif self.hparams.task == 'protein':
            x_train, y_train = batch
            logits = self.model(x_train)
            loss = self.criterion(logits.squeeze(), y_train.squeeze())
            mae = F.l1_loss(logits.squeeze(), y_train.squeeze(), reduction='mean').item()

        elif self.hparams.task == 'cosmic':
            img, mask, ignore = batch
            img1, mask1, ignore1 = set_input(img, mask, ignore, self.data_shape)
            logits = self.model(img1).permute(0, 3, 1, 2).contiguous()  # channel on axis 1
            loss = self.criterion(logits * (1 - ignore1), mask1 * (1 - ignore1))
            mae = 0.0

        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        return {
            'loss': loss,
            'MAE': mae,
        }


    def evaluate_full_dataset(
        self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        print(self.rand_arch)

        loss_sum = 0
        error_sum = 0 
        num_batches = 0
        meter = AverageMeter()
        metric = np.zeros(4)

        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                if self.hparams.task == 'pde':
                    input, target = batch
                    num_batches += 1
                    logits = self.model(input)
                    self.y_normalizer.cuda()
                    logits = self.y_normalizer.decode(logits.squeeze())
                    loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1)).item()
                    loss = loss / logits.size(0)
                    loss_sum += loss

                elif self.hparams.task == 'protein':
                    input, target = batch
                    num_batches += 1 
                    logits = self.model(input)
                    loss = self.criterion(logits, target.squeeze())
                    mae = F.l1_loss(logits, target.squeeze(), reduction='mean').item()
                    loss_sum += loss
                    error_sum += mae



                elif self.hparams.task == 'cosmic':
                    img, mask, ignore = set_input(*batch, self.data_shape)
                    logits = self.model(img).permute(0, 3, 1, 2).contiguous()
                    loss = self.criterion(logits * (1 - ignore), mask * (1 - ignore))
                    meter.update(loss, img.shape[0])
                    metric += maskMetric(logits.reshape
                                         (-1, 1, self.data_shape, self.data_shape).detach().cpu().numpy() > 0.5,
                                         mask.cpu().numpy())

            if self.hparams.task == 'cosmic':
                TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]
                TPR = TP / (TP + FN)
                FPR = FP / (FP + TN)

                results_cosmic = {
                        'validation_error': meter.avg,
                        'FPR': FPR,
                        'TPR': TPR,
                        }

                return results_cosmic

            elif self.hparams.task == 'protein':
                results_protein = {
                    'validation_loss': loss_sum / num_batches,
                    'MAE': error_sum / num_batches,

                }


            elif self.hparams.task == 'pde':

                results_pde = {
                    'validation_loss': loss_sum / num_batches,
                }

                return results_pde

            else: 
                return None


