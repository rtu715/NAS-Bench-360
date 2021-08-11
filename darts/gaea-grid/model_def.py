from typing import Any, Dict, Union, Sequence
import os
from collections import namedtuple
import boto3
import os
import json
import tarfile


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

from data import BilevelDataset, BilevelCosmicDataset
#from model_search import Network
from model_search_expansion import Network
from model_eval import DiscretizedNetwork
#from model_eval_expansion import DiscretizedNetwork
from optimizer import EG
from utils import AttrDict, LpLoss, MatReader, UnitGaussianNormalizer, AverageMeter
from utils import LogCoshLoss, calculate_mae, maskMetric, set_input
import utils

from data_utils.protein_io import load_list
from data_utils.protein_gen import PDNetDataset
from data_utils.cosmic_dataset import PairedDatasetImagePath, get_dirs

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
            #self.criterion = LogCoshLoss()
            self.criterion = nn.MSELoss(reduction='mean')
            #error is reported via MAE
            self.error = nn.L1Loss(reduction='sum')
            self.in_channels = 57
            self.n_classes = 1

        elif self.hparams.task == 'cosmic':
            self.criterion = nn.BCEWithLogitsLoss()
            self.in_channels = 1
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

            self.test_loader = None


        else:
            if self.hparams.task == 'pde':
            
                #newest 5/21 seed 2^31-1 
                searched_genotype = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('max_pool_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
            
                #seed 1
                #searched_genotype = Genotype(normal=[('dil_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
                
                #seed 2
                #searched_genotype = Genotype(normal=[('dil_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_5x5', 1), ('dil_conv_5x5', 4), ('max_pool_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

                #searched_genotype = self.get_genotype_from_hps()

            elif self.hparams.task == 'protein':
                #seed 0
                searched_genotype = Genotype(normal=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 0), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
            
                #seed 1
                #searched_genotype = Genotype(normal=[('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 3), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
            
                #seed 2
                #searched_genotype = Genotype(normal=[('dil_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 0), ('skip_connect', 0), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

            elif self.hparams.task == 'cosmic':
                searched_genotype = Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 2), ('max_pool_3x3', 2), ('max_pool_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6)) 
            
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
                print(x_train.shape)
                print(y_train.shape)

            self.train_data = BilevelDataset(train_data) if self.hparams.train else train_data


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

            if self.hparams.train:
                train_pdbs = self.deepcov_list[100:]

                train_data = PDNetDataset(train_pdbs, self.all_feat_paths, self.all_dist_paths,
                                          128, 10, self.context.get_per_slot_batch_size(), 57,
                                          label_engineering = '16.0')

            else:
                train_pdbs = self.deepcov_list[:]
                train_data = PDNetDataset(train_pdbs, self.all_feat_paths, self.all_dist_paths,
                                          128, 10, self.context.get_per_slot_batch_size(), 57,
                                          label_engineering = '16.0')

            self.train_data = BilevelDataset(train_data) if self.hparams.train else train_data

        elif self.hparams.task == 'cosmic':
            # extract tar file and get directories
            # base_dir = '/workspace/tasks/cosmic/deepCR.ACS-WFC'
            base_dir = self.download_directory
            print(base_dir)
            
            os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
            data_base = os.path.join(base_dir, 'data')
            train_tar = tarfile.open(os.path.join(base_dir, 'deepCR.ACS-WFC.train.tar'))
            test_tar = tarfile.open(os.path.join(base_dir, 'deepCR.ACS-WFC.test.tar'))

            #train_tar.extractall(data_base)
            #test_tar.extractall(data_base)
            get_dirs(base_dir, data_base)

            self.train_dirs = np.load(os.path.join(base_dir, 'train_dirs.npy'), allow_pickle=True)
            self.test_dirs = np.load(os.path.join(base_dir, 'test_dirs.npy'), allow_pickle=True)

            aug_sky = (-0.9, 3)

            # only train f435 and GAL flag for now
            print(self.train_dirs[0])
            if self.hparams.train:
                train_data = PairedDatasetImagePath(self.train_dirs[::], aug_sky[0], aug_sky[1], part='train')
            else:
                train_data = PairedDatasetImagePath(self.train_dirs[::], aug_sky[0], aug_sky[1], part='None')
            self.data_shape = train_data[0][0].shape[1]
            print(len(train_data))

            self.train_data = BilevelCosmicDataset(train_data) if self.hparams.train else train_data

        else:
            raise NotImplementedError

        train_queue = DataLoader(
            self.train_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=2,
        )
        return train_queue

    def build_validation_data_loader(self) -> DataLoader:
        
        protein_test = False
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
                TEST_PATH = os.path.join(self.download_directory, 'piececonst_r421_N1024_smooth2.mat')
                reader = MatReader(TEST_PATH)
                x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]
                y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]

                x_test = self.x_normalizer.encode(x_test)
                x_test = torch.cat([x_test.reshape(ntest, s, s, 1), self.grid.repeat(ntest, 1, 1, 1)], dim=3)

            return DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                          batch_size=self.context.get_per_slot_batch_size(), shuffle=False, num_workers=2,)

        elif self.hparams.task == 'protein':
            if self.hparams.train:
                valid_pdbs = self.deepcov_list[:100]
                valid_data = PDNetDataset(valid_pdbs, self.all_feat_paths, self.all_dist_paths,
                                          128, 10, self.context.get_per_slot_batch_size(), 57,
                                          label_engineering = '16.0')
                valid_queue = DataLoader(valid_data, batch_size=2, shuffle=True, num_workers=2)


            else:
                psicov_list = load_list('psicov.lst')
                psicov_length_dict = {}
                for pdb in psicov_list:
                    (ly, seqy, cb_map) = np.load('psicov/distance/' + pdb + '-cb.npy',
                                                 allow_pickle=True)
                    psicov_length_dict[pdb] = ly

                self.my_list = psicov_list
                self.length_dict = psicov_length_dict

                #note, when testing batch size should be different
                test_data = PDNetDataset(self.my_list, self.all_feat_paths, self.all_dist_paths,
                                         512, 10, 1, 57, label_engineering = None)
                valid_queue = DataLoader(test_data, batch_size=2, shuffle=True, num_workers=0)

            return valid_queue

        elif self.hparams.task == 'cosmic':
            aug_sky = (-0.9,3)
            if self.hparams.train:
                valid_data = PairedDatasetImagePath(self.train_dirs[::], aug_sky[0], aug_sky[1], part='test')
            else:
                valid_data = PairedDatasetImagePath(self.test_dirs[::], aug_sky[0], aug_sky[1], part=None)

            print(len(valid_data))

            valid_queue = DataLoader(valid_data, batch_size=self.context.get_per_slot_batch_size(), shuffle=False, num_workers=8)

            return valid_queue

        else:
            print('no such dataset')
            raise NotImplementedError

        return None


    def build_test_data_loader(self) -> DataLoader:
        batch_size = self.context.get_per_slot_batch_size()

        if self.hparams.task == 'pde':
            ntest = 100
            s = self.s
            r = self.hparams["sub"]

            TEST_PATH = os.path.join(self.download_directory, 'piececonst_r421_N1024_smooth2.mat')
            reader = MatReader(TEST_PATH)
            x_test = reader.read_field('coeff')[:ntest, ::r, ::r][:, :s, :s]
            y_test = reader.read_field('sol')[:ntest, ::r, ::r][:, :s, :s]

            x_test = self.x_normalizer.encode(x_test)
            x_test = torch.cat([x_test.reshape(ntest, s, s, 1), self.grid.repeat(ntest, 1, 1, 1)], dim=3)

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

        elif self.hparams.task == 'cosmic':
            aug_sky = (-0.9,3)
            test_data = PairedDatasetImagePath(self.test_dirs[::], aug_sky[0], aug_sky[1], part=None)
            test_queue = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

        else:
            raise NotImplementedError

        return test_queue

    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:

        if self.hparams.train:
            if epoch_idx != self.last_epoch:
                self.train_data.shuffle_val_inds()
            self.last_epoch = epoch_idx

            if self.hparams.task == 'cosmic':
                img1, mask1, ignore1, img2, mask2, ignore2 = batch
            else:
                x_train, y_train, x_val, y_val = batch
        
            if self.test_loader == None:
                #build test dataloader to eval supernet
                self.test_loader = self.build_test_data_loader()
        
        else:
            if self.hparams.task == 'cosmic':
                img, mask, ignore = batch
            else:
                x_train, y_train = batch
            self.model.drop_path_prob = self.context.get_hparam("drop_path_prob") * epoch_idx / 600.0
            #print('current drop prob is {}'.format(self.model.drop_path_prob))
        
        batch_size = self.context.get_per_slot_batch_size()

        if self.hparams.train:
            # Train shared-weights
            for a in self.model.arch_parameters():
                a.requires_grad = False
            for w in self.model.ws_parameters():
                w.requires_grad = True

            if self.hparams.task =='pde':
                logits = self.model(x_train)
                logits = logits.squeeze()
                self.y_normalizer.cuda()
                target = self.y_normalizer.decode(y_train)
                logits = self.y_normalizer.decode(logits)
                loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1))
                mae = 0

            elif self.hparams.task == 'protein':
                logits = self.model(x_train)
                loss = self.criterion(logits.squeeze(), y_train.squeeze())
                mae = F.l1_loss(logits.squeeze(), y_train.squeeze(), reduction='mean').item()

            elif self.hparams.task == 'cosmic':
                img1, mask1, ignore1 = set_input(img1, mask1, ignore1, self.data_shape)
                logits = self.model(img1).permute(0, 3, 1, 2).contiguous() #channel on axis 1
                loss = self.criterion(logits * (1 - ignore1), mask1 * (1 - ignore1))
                mae = 0.0

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

                if self.hparams.task =='pde':
                    logits = self.model(x_val)
                    logits = logits.squeeze()
                    target = self.y_normalizer.decode(y_val)
                    logits = self.y_normalizer.decode(logits)
                    arch_loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1))

                elif self.hparams.task =='protein':
                    logits = self.model(x_val)
                    arch_loss = self.criterion(logits.squeeze(), y_val.squeeze())

                elif self.hparams.task == 'cosmic':
                    img2, mask2, ignore2 = set_input(img2, mask2, ignore2, self.data_shape)
                    logits = self.model(img2).permute(0, 3, 1, 2).contiguous()
                    arch_loss = self.criterion(logits * (1 - ignore2), mask2 * (1 - ignore2))

                self.context.backward(arch_loss)
                self.context.step_optimizer(self.arch_opt)

        else: 
            if self.hparams.task =='pde':
                self.y_normalizer.cuda()
                logits = self.model(x_train)
                logits = logits.squeeze()
                target = self.y_normalizer.decode(y_train)
                logits = self.y_normalizer.decode(logits)
                loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1))
                mae = 0.0

            elif self.hparams.task =='protein':
                logits = self.model(x_train)
                loss = self.criterion(logits.squeeze(), y_train.squeeze())
                mae = F.l1_loss(logits.squeeze(), y_train.squeeze(), reduction='mean').item()

            elif self.hparams.task == 'cosmic':
                img, mask, ignore = set_input(img, mask, ignore, self.data_shape)
                logits = self.model(img).permute(0, 3, 1, 2).contiguous()
                loss = self.criterion(logits * (1 - ignore), mask * (1 - ignore))
                mae = 0.0
            
            self.context.backward(loss)
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
            "MAE": mae,
        }

    def evaluate_full_dataset(
        self, data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Any]:

        #evaluate protein if eval
        if not self.hparams.train and self.hparams.task=='protein':
            return self.evaluate_test_protein(data_loader)

        loss_sum = 0
        error_sum = 0
        num_batches = 0

        #for cr
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
                    logits = logits.squeeze()
                    logits = self.y_normalizer.decode(logits)
                    loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1)).item()
                    loss = loss / logits.size(0)

                elif self.hparams.task == 'protein':
                    input, target = batch
                    num_batches += 1
                    logits = self.model(input)
                    logits = logits.squeeze()
                    target = target.squeeze()
                    loss = self.criterion(logits, target).item()
                    error = F.l1_loss(logits, target, reduction='mean')
                    error_sum += error.item()

                elif self.hparams.task == 'cosmic':
                    img, mask, ignore = set_input(*batch, self.data_shape)
                    logits = self.model(img).permute(0,3,1,2).contiguous()
                    loss = self.criterion(logits*(1-ignore), mask*(1-ignore))
                    meter.update(loss, img.shape[0])
                    metric += maskMetric(logits.reshape
                                         (-1, 1, self.data_shape, self.data_shape).detach().cpu().numpy() > 0.5, mask.cpu().numpy())
                loss_sum += loss

            if self.hparams.task == 'cosmic':
                TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]
                TPR = TP / (TP + FN)
                FPR = FP / (FP + TN)

                results_cosmic = {'validation_error': meter.avg,
                        'FPR': FPR,
                        'TPR': TPR,
                        }


        test_loss_sum = 0
        test_error_sum = 0
        test_num_batches = 100

        test_meter = AverageMeter()
        test_metric = np.zeros(4)
        
        if self.hparams.train and self.hparams.task=='pde':
            test_num_batches = 0 
            with torch.no_grad():
                for batch in self.test_loader:
                    batch = self.context.to_device(batch)
                    input, target = batch
                    test_num_batches += 1
                    logits = self.model(input)
                    self.y_normalizer.cuda()
                    logits = logits.squeeze()
                    logits = self.y_normalizer.decode(logits)
                    loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1)).item()
                    loss = loss / logits.size(0)
                    error = 0 

                    test_loss_sum += loss
                    test_error_sum += error

        if self.hparams.task == 'cosmic':
            if self.hparams.train:
                with torch.no_grad():
                    for batch in self.test_loader:
                        img, mask, ignore = set_input(*batch, self.data_shape)
                        logits = self.model(img).permute(0,3,1,2).contiguous()
                        loss = self.criterion(logits*(1-ignore), mask*(1-ignore))
                        test_meter.update(loss, img.shape[0])
                        test_metric += maskMetric(logits.reshape
                                             (-1, 1, self.data_shape, self.data_shape).detach().cpu().numpy() > 0.5, mask.cpu().numpy())


                TP, TN, FP, FN = test_metric[0], test_metric[1], test_metric[2], test_metric[3]
                test_TPR = TP / (TP + FN)
                test_FPR = FP / (FP + TN)

                results_cosmic_test = {
                    'test_error': test_meter.avg,
                    'test_TPR': test_TPR,
                    'test_FPR': test_FPR,
                }

                results_cosmic.update(results_cosmic_test)
                

            
            return results_cosmic


        results = {
            "validation_error": loss_sum / num_batches,
            "MAE": error_sum / num_batches,
            "test_error": test_loss_sum / test_num_batches,
            "test_MAE": test_error_sum / test_num_batches,
        }
        
        if self.hparams.train and self.hparams.task =='protein':
            maes = self.evaluate_test_protein(self.test_loader)
            results.update(maes)

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

                #although last layer is linear, out is still shaped bs*1*512*512
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
            'lr8': lr8,
            'mlr8': mlr8,
            'mae12': lr12,
            'mlr12': mlr12,
        }


    def build_callbacks(self):
        return {"genotype": GenotypeCallback(self.context)}
