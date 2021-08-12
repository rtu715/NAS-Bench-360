import argparse
import ast
import importlib
import logging
import os
import pprint
import sys
import time
import boto3
from typing import Any, Dict, Sequence, Tuple, Union, cast
#sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

from determined.pytorch import (
    PyTorchTrial,
    PyTorchTrialContext,
    DataLoader,
    LRScheduler,
    PyTorchCallback
)

from configs.search_config import search_cfg
from configs.imagenet_train_cfg import cfg
from models import model_derived
from models.dropped_model import Dropped_Network
from tools import utils
from tools.config_yaml import merge_cfg_from_file, update_cfg_from_cfg
from tools.lr_scheduler import get_lr_scheduler
from tools.multadds_count import comp_multadds
from data_utils.load_data import load_data
from data_utils.download_data import download_from_s3
#from .optimizer import Optimizer
#from .trainer import SearchTrainer
from data import BilevelDataset

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
        if self.hparams.task == 'ECG':
            merge_cfg_from_file('configs/ecg_search_cfg_resnet.yaml', cfg)
            input_shape = (1, 1000)
            self.criterion = nn.CrossEntropyLoss()

        elif self.hparams.task == 'satellite':
            merge_cfg_from_file('configs/satellite_search_cfg_resnet.yaml', cfg)
            #input_shape = (1, 46)
            input_shape = (1, 48)
            self.criterion = nn.CrossEntropyLoss()


        elif task == 'deepsea':
            merge_cfg_from_file('configs/deepsea_search_cfg_resnet.yaml', cfg)
            input_shape = (4, 1000)
            self.criterion = nn.BCEWithLogitsLoss()

        else:
            raise NotImplementedError
        
        config = cfg
        self.input_shape = input_shape
        pprint.pformat(config)
        
        cudnn.benchmark = True
        cudnn.enabled = True
        self.criterion = self.criterion.cuda()

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

        flops_list, total_flops = super_model.get_cost_list(
                                input_shape, cost_type='flops')
        super_model.sub_obj_list = flops_list
        print("Super Network flops (M) list: \n")
        print(str(flops_list))
        print("Total flops: " + str(total_flops))

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


        scheduler = get_lr_scheduler(config, self.weight_optimizer, self.hparams.num_examples, self.context.get_per_slot_batch_size())

        scheduler.last_step = 0
        self.scheduler = self.context.wrap_lr_scheduler(scheduler, step_mode=LRScheduler.StepMode.MANUAL_STEP)

        self.config = config 
        self.download_directory = self.download_data_from_s3()

    def download_data_from_s3(self):
        '''Download data from s3 to store in temp directory'''

        s3_bucket = self.context.get_data_config()["bucket"]
        download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"
        s3 = boto3.client("s3")
        os.makedirs(download_directory, exist_ok=True)
        #download_directory = '.'

        download_from_s3(s3_bucket, self.hparams.task, download_directory)

        self.train_data, self.val_data, _ = load_data(self.hparams.task, download_directory, True)

        return download_directory

    def build_training_data_loader(self) -> DataLoader:

        self.train_data = BilevelDataset(self.train_data)
        print('Length of bilevel dataset: ', len(self.train_data))

        return DataLoader(self.train_data, batch_size=self.context.get_per_slot_batch_size(), shuffle=True, num_workers=2)

    def build_validation_data_loader(self) -> DataLoader:

        valset = self.val_data

        return DataLoader(valset, batch_size=self.context.get_per_slot_batch_size(), shuffle=False, num_workers=2)

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        
        if epoch_idx != self.last_epoch:
            self.train_data.shuffle_val_inds()
        self.last_epoch = epoch_idx
        
        search_stage = 1 if epoch_idx > self.config.search_params.arch_update_epoch else 0 
        x_train1, y_train1, x_train2, y_train2, x_train3, y_train3, x_train4, y_train4, x_val, y_val = batch
        x_train = torch.cat((x_train1, x_train2, x_train3, x_train4), 0)
        y_train = torch.cat((y_train1, y_train2, y_train3, y_train4), 0)
        
        #n = x_train1.size(0) * 4
        #print('batch size is: ', n)
        arch_loss = 0
        if search_stage:
            self.set_param_grad_state('Arch')
            arch_logits, arch_loss, arch_subobj = self.arch_step(x_val, y_val, self.model, search_stage)
        
        self.scheduler.step()
        self.set_param_grad_state('Weights')
        logits, loss, subobj = self.weight_step(x_train, y_train, self.model, search_stage)

        return {
                'loss': loss,
                'arch_loss': arch_loss,
                }

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

    def evaluate_full_dataset_ECG(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:

        obj = utils.AverageMeter()
        sub_obj = utils.AverageMeter()
        all_pred_prob = []

        self.set_param_grad_state('')
        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                n = input.size(0)
                logits, loss, subobj = self.valid_step(input, target, self.model)
                obj.update(loss, n)
                sub_obj.update(subobj, n)
                all_pred_prob.append(logits.cpu().data.numpy())

        '''for ecg validation'''
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
            'validation_loss': obj.avg,
            'validation_subloss': sub_obj.avg,
            'score': f1_score,
        }

    def evaluate_full_dataset_satellite(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        
        obj = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        sub_obj = utils.AverageMeter()

        self.set_param_grad_state('')
        with torch.no_grad():
            for batch in data_loader:
                batch = self.context.to_device(batch)
                input, target = batch
                n = input.size(0)
                logits, loss, subobj = self.valid_step(input, target, self.model)
                prec1, prec5 = utils.accuracy(logits, target, topk=(1,5))
                obj.update(loss, n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)
                sub_obj.update(subobj, n)

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
                'validation_loss': obj.avg,
                'validation_subloss': sub_obj.avg,
                'validation_accuracy': top1.avg,
                'validation_top5': top5.avg
                }

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
                logits, loss, subobj = self.valid_step(input, target, self.model)
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
        

    def weight_step(self, input_train, target_train, model, search_stage):
        _, _ = model.sample_branch('head', self.weight_sample_num, search_stage=search_stage)
        _, _ = model.sample_branch('stack', self.weight_sample_num, search_stage=search_stage)

        self.weight_optimizer.zero_grad()
        dropped_model = self.Dropped_Network(model)
        logits, sub_obj = dropped_model(input_train)
        sub_obj = torch.mean(sub_obj)
        loss = self.criterion(logits, target_train)
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
        loss = self.criterion(logits, target_valid)
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
        loss = self.criterion(logits, target_valid)

        return logits, loss.item(), sub_obj.item()
