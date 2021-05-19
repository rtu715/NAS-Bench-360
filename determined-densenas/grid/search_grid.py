import importlib
import os
import pprint
import boto3
from typing import Any, Dict, Sequence, Union

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
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
from tools.multadds_count import comp_multadds
from data import BilevelDataset
from utils_grid import LpLoss, MatReader, UnitGaussianNormalizer, LogCoshLoss
from utils_grid import create_grid, filter_MAE

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
            input_shape = (57, 64, 64)
            self.criterion = LogCoshLoss()
            #error is reported via MAE
            self.error = nn.L1Loss(reduction='sum')
            self.in_channels = 57

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


        scheduler = CosineAnnealingLR(self.weight_optimizer, config.train_params.epochs, config.optim.min_lr)
        self.scheduler = self.context.wrap_lr_scheduler(scheduler, step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH)

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
            data_files = ['X_train.npz', 'X_valid.npz', 'Y_train.npz', 'Y_valid.npz']
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
        if self.hparams.task == 'pde':
            TRAIN_PATH = os.path.join(self.download_directory, 'piececonst_r421_N1024_smooth1.mat')
            self.reader = MatReader(TRAIN_PATH)
            s = self.s
            r = self.hparams["sub"]
            ntrain = 1000
            ntest = 100
            if self.hparams.train:
                x_train = self.reader.read_field('coeff')[:ntrain - ntest, ::r, ::r][:, :s, :s]
                y_train = self.reader.read_field('sol')[:ntrain - ntest, ::r, ::r][:, :s, :s]

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
                # save 100 samples from train set as validation
                x_train = x_train[100:]
                y_train = y_train[100:]

            train_data = torch.utils.data.TensorDataset(x_train, y_train)


        self.train_data = BilevelDataset(train_data)
        train_queue = DataLoader(
            self.train_data,
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

            if self.hparams.train:
                x_test = self.reader.read_field('coeff')[ntrain - ntest:ntrain, ::r, ::r][:, :s, :s]
                y_test = self.reader.read_field('sol')[ntrain - ntest:ntrain, ::r, ::r][:, :s, :s]

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
        return DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                          batch_size=self.context.get_per_slot_batch_size(), shuffle=False, num_workers=2,)

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
    
        if epoch_idx != self.last_epoch:
            self.train_data.shuffle_val_inds()
        self.last_epoch = epoch_idx
        
        search_stage = 1 if epoch_idx > self.config.search_params.arch_update_epoch else 0 

        #x_train, y_train, x_val, y_val = batch

        x_train1, y_train1, x_train2, y_train2, x_train3, y_train3, x_train4, y_train4, x_val, y_val = batch
        x_train = torch.cat((x_train1, x_train2, x_train3, x_train4), 0)
        y_train = torch.cat((y_train1, y_train2, y_train3, y_train4), 0)        

        arch_loss = 0
        if search_stage:
            self.set_param_grad_state('Arch')
            arch_logits, arch_loss, arch_subobj = self.arch_step(x_val, y_val, self.model, search_stage)
        
        self.set_param_grad_state('Weights')
        logits, loss, subobj = self.weight_step(x_train, y_train, self.model, search_stage)

        return {
                'loss': loss,
                'arch_loss': arch_loss,
                }


    def evaluate_full_dataset(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:

        obj = 0.0
        sub_obj = 0.0
        error_sum = 0.0
        num_batches = 0
        self.set_param_grad_state('')
        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                logits, loss, subobj, error = self.valid_step(input, target, self.model)

                num_batches += 1
                obj += loss
                sub_obj += subobj
                error_sum += error

        betas, head_alphas, stack_alphas = self.model.display_arch_params()
        derived_arch = self.arch_gener.derive_archs(betas, head_alphas, stack_alphas)
        derived_arch_str = '|\n'.join(map(str, derived_arch))
        derived_model = self.der_Net(derived_arch_str)
        derived_flops = comp_multadds(derived_model, input_size=self.input_shape)
        derived_params = utils.count_parameters_in_MB(derived_model)
        print("Derived Model Mult-Adds = %.2fMB" % derived_flops)
        print("Derived Model Num Params = %.2fMB" % derived_params)
        print(derived_arch_str)

        return {
                'validation_loss': obj / num_batches,
                'validation_subloss': sub_obj / num_batches,
                'MAE': error_sum / num_batches,
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

        elif self.hparams.task == 'protein':
            loss = self.criterion(logits, target_train.squeeze())

        loss.backward()
        self.weight_optimizer.step()

        return logits.detach(), loss.item(), sub_obj.item()

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
            target = self.y_normalizer.decode(target_valid)
            logits = self.y_normalizer.decode(logits)
            loss = self.criterion(logits.view(logits.size(0), -1), target.view(target.size(0), -1))
            loss = loss / logits.size(0)
            error = 0

        elif self.hparams.task == 'protein':
            loss = self.criterion(logits, target_valid.squeeze())
            loss = loss / logits.size(0)

            target_valid, logits, num = filter_MAE(target_valid, logits, 8.0)
            error = self.error(logits, target_valid)
            error = error / num

        return logits, loss.item(), sub_obj.item(), error
