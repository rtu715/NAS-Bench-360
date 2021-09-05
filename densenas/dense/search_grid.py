import importlib
import os
import pprint
import boto3
import json
from typing import Any, Dict, Sequence, Union
import tarfile

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from determined.pytorch import (
    PyTorchTrial,
    PyTorchTrialContext,
    DataLoader,
    LRScheduler
)

from configs.search_config import search_cfg
from configs.imagenet_train_cfg import cfg
from models import model_derived
from models.dropped_model import Dropped_Network
from tools import utils
from tools.config_yaml import merge_cfg_from_file, update_cfg_from_cfg
from tools.lr_scheduler import get_lr_scheduler
from tools.multadds_count import comp_multadds
from data import BilevelDataset, BilevelCosmicDataset
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

        update_cfg_from_cfg(search_cfg, cfg)
        if self.hparams.task == 'pde':
            merge_cfg_from_file('configs/pde_search_cfg_resnet.yaml', cfg)
            input_shape = (3, 85, 85)
            self.grid, self.s = create_grid(self.hparams.sub)
            self.criterion = LpLoss(size_average=False)
            self.in_channels = 3

        elif self.hparams.task == 'protein':
            merge_cfg_from_file('configs/protein_search_cfg_resnet.yaml', cfg)
            input_shape = (57, 128, 128)
            #self.criterion = LogCoshLoss()
            self.criterion = nn.MSELoss(reduction='mean')
            #error is reported via MAE
            self.error = nn.L1Loss(reduction='sum')
            self.in_channels = 57

        elif self.hparams.task == 'cosmic':
            merge_cfg_from_file('configs/cosmic_search_cfg_resnet.yaml', cfg)
            input_shape = (1, 256, 256)
            self.criterion = nn.BCEWithLogitsLoss()
            self.in_channels = 1

        else:
            raise NotImplementedError
        
        config = cfg
        self.input_shape = input_shape
        pprint.pformat(config)
        
        cudnn.benchmark = True
        cudnn.enabled = True

        SearchSpace = importlib.import_module('models.search_space_'+self.hparams.net_type).Network
        ArchGenerater = importlib.import_module('run_apis.derive_arch_'+self.hparams.net_type, __package__).ArchGenerate
        derivedNetwork = getattr(model_derived, '%s_Net' % self.hparams.net_type.upper())

        super_model = SearchSpace(config.optim.init_dim, self.hparams.task, config)
        self.arch_gener = ArchGenerater(super_model, config)
        self.der_Net = lambda net_config: derivedNetwork(net_config, task=self.hparams.task,
                                                    config=config)
        #super_model = nn.DataParallel(super_model)
        #if need to parallel, evaluate batch not full dataet
        super_model = super_model.cuda()

        if config.optim.sub_obj.type=='flops':
            flops_list, total_flops = super_model.get_cost_list(
                                    input_shape, cost_type='flops')
            super_model.sub_obj_list = flops_list
            print("Super Network flops (M) list: \n")
            print(str(flops_list))
            print("Total flops: " + str(total_flops))
            '''
        elif config.optim.sub_obj.type=='latency':
            with open(os.path.join('latency_list', config.optim.sub_obj.latency_list_path), 'r') as f:
                latency_list = eval(f.readline())
            super_model.module.sub_obj_list = latency_list
            print("Super Network latency (ms) list: \n")
            print(str(latency_list))
            '''
        else:
            raise NotImplementedError

        pprint.pformat("Num params = %.2fMB", utils.count_parameters_in_MB(super_model))
        self.model = self.context.wrap_model(super_model)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)/ 1e6
        print('Parameter size in MB: ', total_params)

        self.Dropped_Network = lambda model: Dropped_Network(
                        model, softmax_temp=config.search_params.softmax_temp)

        arch_params_id = list(map(id, self.model.arch_parameters))
        weight_params = filter(lambda p: id(p) not in arch_params_id, self.model.parameters())
        self.weight_sample_num = config.search_params.weight_sample_num


        self.weight_optimizer = self.context.wrap_optimizer(torch.optim.SGD(
                                weight_params,
                                config.optim.weight.init_lr,
                                momentum=config.optim.weight.momentum,
                                weight_decay=config.optim.weight.weight_decay))

        self.arch_optimizer = self.context.wrap_optimizer(torch.optim.Adam(
                            [{'params': self.model.arch_alpha_params, 'lr': config.optim.arch.alpha_lr},
                                {'params': self.model.arch_beta_params, 'lr': config.optim.arch.beta_lr}],
                            betas=(0.5, 0.999),
                            weight_decay=config.optim.arch.weight_decay))


        #scheduler = CosineAnnealingLR(self.weight_optimizer, config.train_params.epochs, config.optim.min_lr)
        scheduler = get_lr_scheduler(config, self.weight_optimizer, self.hparams.num_examples, self.context.get_per_slot_batch_size())
        scheduler.last_step = 0

        self.scheduler = self.context.wrap_lr_scheduler(scheduler, step_mode=LRScheduler.StepMode.MANUAL_STEP)

        self.config = config 
        self.download_directory = self.download_data_from_s3()
        self.test_loader = None

        '''for generating random arch'''
        betas, head_alphas, stack_alphas = self.model.display_arch_params()
        derived_arch = self.arch_gener.derive_archs(betas, head_alphas, stack_alphas)
        derived_arch_str = '|\n'.join(map(str, derived_arch))
        derived_model = self.der_Net(derived_arch_str)
        print(derived_arch_str)

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

            print(x_train.shape)
            print(y_train.shape)

            self.train_data = BilevelDataset(train_data)

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
            self.train_data = BilevelDataset(train_data)

        elif self.hparams.task == 'cosmic':
            # extract tar file and get directories
            #base_dir = '/tmp/data-rank0'
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

            self.train_data = BilevelCosmicDataset(train_data) if self.hparams.train else train_data

        else:
            raise NotImplementedError

        train_queue = DataLoader(
            self.train_data,
            batch_size=self.context.get_per_slot_batch_size(),
            shuffle=True,
            num_workers=8,
        )
        return train_queue

    def build_validation_data_loader(self) -> DataLoader:

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

        #build test loader
        self.test_loader = self.build_test_data_loader()
        
        return valid_queue


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

        
    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
    
        if epoch_idx != self.last_epoch:
            self.train_data.shuffle_val_inds()
        self.last_epoch = epoch_idx

        #if epoch_idx == 0 and self.test_loader == None:
        
        search_stage = 1 if epoch_idx > self.config.search_params.arch_update_epoch else 0

        if self.hparams.task == 'cosmic':
            img1, mask1, ignore1, img2, mask2, ignore2, img3, mask3, ignore3,\
                img4, mask4, ignore4, img_val, mask_val, ignore_val= batch
            train_img = torch.cat((img1, img2, img3, img4), 0)
            train_mask = torch.cat((mask1, mask2, mask3, mask4), 0)
            train_ignore = torch.cat((ignore1, ignore2, ignore3, ignore4), 0)

        else:
            x_train1, y_train1, x_train2, y_train2, x_train3, y_train3, x_train4, y_train4, x_val, y_val = batch
            x_train = torch.cat((x_train1, x_train2, x_train3, x_train4), 0)
            y_train = torch.cat((y_train1, y_train2, y_train3, y_train4), 0)

        arch_loss = 0
        if search_stage:
            self.set_param_grad_state('Arch')
            if self.hparams.task == 'cosmic':
                arch_logits, arch_loss, arch_subobj = self.arch_step_cosmic(train_img, train_mask, train_ignore,
                                                                            self.model, search_stage)
            else:
                arch_logits, arch_loss, arch_subobj = self.arch_step(x_val, y_val, self.model, search_stage)

        self.scheduler.step()
        self.set_param_grad_state('Weights')
        if self.hparams.task == 'cosmic':
            logits, loss, subobj, mae = self.weight_step_cosmic(train_img, train_mask, train_ignore, self.model, search_stage)
        else:
            logits, loss, subobj, mae = self.weight_step(x_train, y_train, self.model, search_stage)

        return {
                'loss': loss,
                'arch_loss': arch_loss,
                'MAE': mae,
                }


    def evaluate_full_dataset(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:

        obj = 0.0
        sub_obj = 0.0
        error_sum = 0.0
        num_batches = 0

        #for cr
        meter = AverageMeter()
        metric = np.zeros(4)

        self.set_param_grad_state('')
        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                if self.hparams.task == 'cosmic':
                    logits, loss, subobj = self.valid_step_cosmic(*batch, self.model)
                    _, mask, _ = set_input(*batch, self.data_shape)
                    meter.update(loss, mask.shape[0])
                    metric += maskMetric(logits.reshape
                                         (-1, 1, self.data_shape, self.data_shape).detach().cpu().numpy() > 0.5,
                                         mask.cpu().numpy())

                else:
                    input, target = batch
                    logits, loss, subobj, error = self.valid_step(input, target, self.model)

                    num_batches += 1
                    obj += loss
                    sub_obj += subobj
                    error_sum += error

        test_obj = 0.0
        test_sub_obj = 0.0
        test_error_sum = 0.0
        test_num_batches = 100
        lr8, mlr8, lr12, mlr12 = 0.0, 0.0, 0.0, 0.0

        test_meter = AverageMeter()
        test_metric = np.zeros(4)

        if self.hparams.task == 'pde':
            test_num_batches = 0
            with torch.no_grad():
                for batch in self.test_loader:
                    batch = self.context.to_device(batch)
                    input, target = batch
                    logits, loss, subobj, error = self.valid_step(input, target, self.model)
                    test_num_batches += 1
                    test_obj += loss
                    test_sub_obj += subobj
                    test_error_sum += error

        elif self.hparams.task == 'protein':
            LMAX = 512  # psicov constant
            L = 128
            pad_size = 10
            dropped_model = self.Dropped_Network(self.model)
            with torch.no_grad():
                P = []
                targets = []
                for batch in self.test_loader:
                    batch = self.context.to_device(batch)
                    data, target = batch
                    for i in range(data.size(0)):
                        targets.append(
                            np.expand_dims(
                                target.cpu().numpy()[i].transpose(1, 2, 0), axis=0))

                    _, _, s_length, _ = data.shape
                    stride = L
                    y = torch.zeros_like(data)[:, :1, :, :]
                    counts = torch.zeros_like(data)[:, :1, :, :]
                    for i in range((((s_length - L) // stride)) + 1):
                        ip = i * stride
                        for j in range((((s_length - L) // stride)) + 1):
                            jp = j * stride
                            logits, _ = dropped_model(data[:, :, ip:ip + L, jp:jp + L])
                            logits = logits.unsqueeze(3).permute(0,3,1,2)
                            y[:, :, ip:ip + L, jp:jp + L] = logits
                            counts[:, :, ip:ip + L, jp:jp + L] += torch.ones_like(logits)

                    out = y / counts
                    #transpose here since out is [bs,1, size, size]
                    P.append(out.cpu().numpy().transpose(0, 2, 3, 1))

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

        elif self.hparams.task == 'cosmic':
            with torch.no_grad():
                for batch in self.test_loader:
                    logits, loss, subobj = self.valid_step_cosmic(*batch, self.model)
                    _, mask, _ = set_input(*batch, self.data_shape)
                    test_meter.update(loss, mask.shape[0])
                    test_metric += maskMetric(logits.reshape
                                         (-1, 1, self.data_shape, self.data_shape).detach().cpu().numpy() > 0.5,
                                              mask.cpu().numpy())



        betas, head_alphas, stack_alphas = self.model.display_arch_params()
        derived_arch = self.arch_gener.derive_archs(betas, head_alphas, stack_alphas)
        derived_arch_str = '|\n'.join(map(str, derived_arch))
        derived_model = self.der_Net(derived_arch_str)
        derived_flops = comp_multadds(derived_model, input_size=self.input_shape)
        derived_params = utils.count_parameters_in_MB(derived_model)
        print("Derived Model Mult-Adds = %.2fMB" % derived_flops)
        print("Derived Model Num Params = %.2fMB" % derived_params)
        print(derived_arch_str)
        #print('num batches is: ', num_batches)

        if self.hparams.task == 'cosmic':
            TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)

            results_cosmic = {'validation_error': meter.avg,
                              'FPR': FPR,
                              'TPR': TPR,
                              }

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

        return {
                'validation_loss': obj / num_batches,
                'validation_subloss': sub_obj / num_batches,
                'MAE': error_sum / num_batches,
                'test_loss': test_obj / test_num_batches,
                'test_subloss': test_sub_obj / test_num_batches,
                'test_MAE': test_error_sum / test_num_batches,
                'lr8': lr8,
                'lr12': lr12,
                'mlr8': mlr8,
                'mlr12': mlr12,
                }
        

    def weight_step(self, input_train, target_train, model, search_stage):
        _, _ = model.sample_branch('head', self.weight_sample_num, search_stage=search_stage)
        _, _ = model.sample_branch('stack', self.weight_sample_num, search_stage=search_stage)

        self.weight_optimizer.zero_grad()
        dropped_model = self.Dropped_Network(model)
        logits, sub_obj = dropped_model(input_train)
        sub_obj = torch.mean(sub_obj)
        if self.hparams.task == 'pde':
            self.y_normalizer.cuda()
            target = self.y_normalizer.decode(target_train)
            logits = self.y_normalizer.decode(logits)
            loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1))
            mae = 0.0
        elif self.hparams.task == 'protein':
            loss = self.criterion(logits, target_train.squeeze())
            mae = F.l1_loss(logits, target_train.squeeze(), reduction='mean').item()

        loss.backward()
        self.weight_optimizer.step()

        return logits.detach(), loss.item(), sub_obj.item(), mae

    def weight_step_cosmic(self, img, mask, ignore, model, search_stage):
        _, _ = model.sample_branch('head', self.weight_sample_num, search_stage=search_stage)
        _, _ = model.sample_branch('stack', self.weight_sample_num, search_stage=search_stage)

        self.weight_optimizer.zero_grad()
        dropped_model = self.Dropped_Network(model)
        img, mask, ignore = set_input(img, mask, ignore, self.data_shape)
        logits, sub_obj = dropped_model(img)
        sub_obj = torch.mean(sub_obj)
        logits = logits.reshape(-1, 1, self.data_shape, self.data_shape)
        loss = self.criterion(logits * (1 - ignore), mask * (1 - ignore))
        mae = 0.0
        loss.backward()
        self.weight_optimizer.step()

        return logits.detach(), loss.item(), sub_obj.item(), mae


    def set_param_grad_state(self, stage):
        def set_grad_state(params, state):
            for group in params:
                for param in group['params']:
                    param.requires_grad_(state)
        if stage == 'Arch':
            state_list = [True, False] # [arch, weight]
        elif stage == 'Weights':
            state_list = [False, True]
        else:
            state_list = [False, False]
        set_grad_state(self.arch_optimizer.param_groups, state_list[0])
        set_grad_state(self.weight_optimizer.param_groups, state_list[1])

    def arch_step(self, input_valid, target_valid, model, search_stage):
        head_sampled_w_old, alpha_head_index = \
            model.sample_branch('head', 2, search_stage= search_stage)
        stack_sampled_w_old, alpha_stack_index = \
            model.sample_branch('stack', 2, search_stage= search_stage)
        self.arch_optimizer.zero_grad()

        dropped_model = self.Dropped_Network(model)
        logits, sub_obj = dropped_model(input_valid)
        sub_obj = torch.mean(sub_obj)

        if self.hparams.task == 'pde':
            self.y_normalizer.cuda()
            target = self.y_normalizer.decode(target_valid)
            logits = self.y_normalizer.decode(logits)
            loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1))

        elif self.hparams.task == 'protein':
            loss = self.criterion(logits, target_valid.squeeze())

        if self.config.optim.if_sub_obj:
            loss_sub_obj = torch.log(sub_obj) / torch.log(torch.tensor(self.config.optim.sub_obj.log_base))
            sub_loss_factor = self.config.optim.sub_obj.sub_loss_factor
            loss += loss_sub_obj * sub_loss_factor

        loss.backward()
        self.arch_optimizer.step()

        self.rescale_arch_params(head_sampled_w_old,
                                stack_sampled_w_old,
                                alpha_head_index,
                                alpha_stack_index,
                                model)
        return logits.detach(), loss.item(), sub_obj.item()

    def arch_step_cosmic(self, img_valid, mask_valid, ignore_valid, model, search_stage):
        head_sampled_w_old, alpha_head_index = \
            model.sample_branch('head', 2, search_stage= search_stage)
        stack_sampled_w_old, alpha_stack_index = \
            model.sample_branch('stack', 2, search_stage= search_stage)
        self.arch_optimizer.zero_grad()

        img, mask, ignore = set_input(img_valid, mask_valid, ignore_valid, self.data_shape)
        dropped_model = self.Dropped_Network(model)
        logits, sub_obj = dropped_model(img)
        #expand dim for logits
        logits = logits.reshape(-1, 1, self.data_shape, self.data_shape)
        loss = self.criterion(logits * (1 - ignore), mask * (1 - ignore))
        sub_obj = torch.mean(sub_obj)

        if self.config.optim.if_sub_obj:
            loss_sub_obj = torch.log(sub_obj) / torch.log(torch.tensor(self.config.optim.sub_obj.log_base))
            sub_loss_factor = self.config.optim.sub_obj.sub_loss_factor
            loss += loss_sub_obj * sub_loss_factor

        loss.backward()
        self.arch_optimizer.step()

        self.rescale_arch_params(head_sampled_w_old,
                                stack_sampled_w_old,
                                alpha_head_index,
                                alpha_stack_index,
                                model)
        return logits.detach(), loss.item(), sub_obj.item()


    def rescale_arch_params(self, alpha_head_weights_drop, 
                            alpha_stack_weights_drop,
                            alpha_head_index,
                            alpha_stack_index,
                            model):

        def comp_rescale_value(old_weights, new_weights, index):
            old_exp_sum = old_weights.exp().sum()
            new_drop_arch_params = torch.gather(new_weights, dim=-1, index=index)
            new_exp_sum = new_drop_arch_params.exp().sum()
            rescale_value = torch.log(old_exp_sum / new_exp_sum).item() 
            rescale_mat = torch.zeros_like(new_weights).scatter_(0, index, rescale_value)
            return rescale_value, rescale_mat
        
        def rescale_params(old_weights, new_weights, indices):
            for i, (old_weights_block, indices_block) in enumerate(zip(old_weights, indices)):
                for j, (old_weights_branch, indices_branch) in enumerate(
                                                    zip(old_weights_block, indices_block)):
                    rescale_value, rescale_mat = comp_rescale_value(old_weights_branch,
                                                                new_weights[i][j],
                                                                indices_branch)
                    new_weights[i][j].data.add_(rescale_mat)

        # rescale the arch params for head layers
        rescale_params(alpha_head_weights_drop, model.alpha_head_weights, alpha_head_index)
        # rescale the arch params for stack layers
        rescale_params(alpha_stack_weights_drop, model.alpha_stack_weights, alpha_stack_index)

    def valid_step(self, input_valid, target_valid, model):
        _, _ = model.sample_branch('head', 1, training=False)
        _, _ = model.sample_branch('stack', 1, training=False)

        dropped_model = self.Dropped_Network(model)
        logits, sub_obj = dropped_model(input_valid)
        sub_obj = torch.mean(sub_obj)
        if self.hparams.task == 'pde':
            self.y_normalizer.cuda()
            logits = self.y_normalizer.decode(logits) 
            
            loss = self.criterion(logits.view(logits.size(0), -1), target_valid.view(target_valid.size(0), -1))
            loss = loss / logits.size(0)
            mae = 0

        elif self.hparams.task == 'protein':
            loss = self.criterion(logits, target_valid.squeeze())
            mae = F.l1_loss(logits, target_valid.squeeze(), reduction='mean').item()
            #target_valid, logits, num = filter_MAE(target_valid.squeeze(), logits.squeeze(), 8.0)
            #error = self.error(logits, target_valid).item()
            #if num and error:
            #    error = error / num
            #else:
            #    error = 0

        return logits, loss.item(), sub_obj.item(), mae

    def valid_step_cosmic(self, img_val, mask_val, ignore_val, model):
        _, _ = model.sample_branch('head', 1, training=False)
        _, _ = model.sample_branch('stack', 1, training=False)
        img, mask, ignore = set_input(img_val, mask_val, ignore_val, self.data_shape)
        dropped_model = self.Dropped_Network(model)
        logits, sub_obj = dropped_model(img)
        sub_obj = torch.mean(sub_obj)

        logits = logits.reshape(-1, 1, self.data_shape, self.data_shape)
        loss = self.criterion(logits * (1 - ignore), mask * (1 - ignore))
        return logits, loss.item(), sub_obj.item()

