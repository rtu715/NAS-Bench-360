import os
import pprint
import sys
import boto3
import json
from typing import Any, Dict, Sequence, Union

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
from models import model_derived
from tools import utils
from utils_grid import LpLoss, MatReader, UnitGaussianNormalizer, LogCoshLoss
from utils_grid import create_grid, calculate_mae
from data_utils.protein_io import load_list
from data_utils.protein_gen import PDNetDataset


TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class DenseNASTrainTrial(PyTorchTrial):

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

        else:
            raise NotImplementedError

        cudnn.benchmark = True
        cudnn.enabled = True

        config.net_config, config.net_type = self.hparams.net_config, self.hparams.net_type
        derivedNetwork = getattr(model_derived, '%s_Net' % self.hparams.net_type.upper())
        model = derivedNetwork(config.net_config, task=self.hparams.task, config=config)


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

            train_pdbs = self.deepcov_list[:]

            train_data = PDNetDataset(train_pdbs, self.all_feat_paths, self.all_dist_paths,
                                      128, 10, self.context.get_per_slot_batch_size(), 57,
                                      label_engineering = '16.0')

        train_queue = DataLoader(
            train_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=2,
        )
        return train_queue

    def build_validation_data_loader(self) -> DataLoader:

        if self.hparams.task == 'pde':
            ntrain = 1000
            ntest = 100
            s = self.s
            r = self.hparams["sub"]

            TEST_PATH = os.path.join(self.download_directory, 'piececonst_r421_N1024_smooth2.mat')
            reader = MatReader(TEST_PATH)
            x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]
            y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]

            x_test = self.x_normalizer.encode(x_test)
            x_test = torch.cat([x_test.reshape(ntest, s, s, 1), self.grid.repeat(ntest, 1, 1, 1)], dim=3)
            batch_size = self.context.get_per_slot_batch_size()

            test_queue = DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                    batch_size=batch_size, shuffle=False, num_workers=2, )

        elif self.hparams.task == 'protein':
            psicov_list = load_list('psicov.lst')
            psicov_length_dict = {}
            for pdb in psicov_list:
                (ly, seqy, cb_map) = np.load('psicov/distance/' + pdb + '-cb.npy',
                                             allow_pickle=True)
                psicov_length_dict[pdb] = ly

            self.my_list = psicov_list
            self.length_dict = psicov_length_dict

            # note, when testing batch size should be different
            test_data = PDNetDataset(self.my_list, self.all_feat_paths, self.all_dist_paths,
                                     512, 10, 1, 57, label_engineering=None)
            test_queue = DataLoader(test_data, batch_size=2, shuffle=True, num_workers=0)

        return test_queue

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int
                    ) -> Dict[str, torch.Tensor]:


        x_train, y_train = batch
        self.scheduler.step()
        self.model.train()
        logits = self.model(x_train)

        if self.hparams.task == 'pde':
            self.y_normalizer.cuda()
            target = self.y_normalizer.decode(y_train.squeeze())
            logits = self.y_normalizer.decode(logits.squeeze())
            loss = self.criterion(logits.view(logits.size(0), -1), target.view(logits.size(0), -1))
            mae = 0.0

        elif self.hparams.task == 'protein':
            loss = self.criterion(logits.squeeze(), y_train.squeeze())
            mae = F.l1_loss(logits.squeeze(), y_train.squeeze(), reduction='mean').item()

        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        return {
            'loss': loss,
            'MAE': mae,
        }


    def evaluate_full_dataset(
        self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:

        if self.hparams.task == 'protein':
            return self.evaluate_test_protein(data_loader)

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
                    self.y_normalizer.cuda()
                    logits = self.y_normalizer.decode(logits.squeeze())
                    loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1)).item()
                    loss = loss / logits.size(0)

                #elif self.hparams.task == 'protein':
                #    print(target.shape)
                #    print(logits.shape)

                #    target = target.squeeze()
                #    logits = logits.squeeze()
                #    loss = self.criterion(logits, target)

                #    mae = F.l1_loss(logits, logits, reduction='mean').item()

                    #target, logits, num = filter_MAE(target, logits, 8.0)
                    #error = self.error(logits, target)
                    #error = error / num
                    #error_sum += mae

                loss_sum += loss

        results = {
            "validation_loss": loss_sum / num_batches,
        }

        return results

    def evaluate_test_protein(
            self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:
        '''performs evaluation on protein'''

        LMAX = 512  # psicov constant
        pad_size = 10
        self.model.cuda()
        with torch.no_grad():
            P = []
            targets = []
            for batch in data_loader:
                batch = self.context.to_device(batch)
                data, target = batch
                for i in range(data.size(0)):
                    targets.append(
                        np.expand_dims(
                            target.cpu().numpy()[i].transpose(1, 2, 0), axis=0))

                out = self.model.forward_window(data, 128)

                P.append(out.cpu().numpy().transpose(0,2,3,1))

            # Combine P, convert to numpy
            P = np.concatenate(P, axis=0)

        Y = np.full((len(targets), LMAX, LMAX, 1), np.nan)
        for i, xy in enumerate(targets):
            Y[i, :, :, 0] = xy[0, :, :, 0]
        # Average the predictions from both triangles
        for j in range(0, len(P[0, :, 0, 0])):
            for k in range(j, len(P[0, :, 0, 0])):
                P[:, j, k, :] = (P[:, k, j, :] + P[:, j, k, :]) / 2.0
        P[P < 0.01] = 0.01

        # Remove padding, i.e. shift up and left by int(pad_size/2)
        P[:, :LMAX - pad_size, :LMAX - pad_size, :] = P[:, int(pad_size / 2): LMAX - int(pad_size / 2),
                                                      int(pad_size / 2): LMAX - int(pad_size / 2), :]
        Y[:, :LMAX - pad_size, :LMAX - pad_size, :] = Y[:, int(pad_size / 2): LMAX - int(pad_size / 2),
                                                      int(pad_size / 2): LMAX - int(pad_size / 2), :]

        print('')
        print('Evaluating distances..')
        lr8, mlr8, lr12, mlr12 = calculate_mae(P, Y, self.my_list, self.length_dict)

        return {
            'mae': lr8,
            'mlr8': mlr8,
            'mae12': lr12,
            'mlr12': mlr12,
        }