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


import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter

from determined.pytorch import (
    PyTorchTrial,
    PyTorchTrialContext,
    DataLoader,
    LRScheduler,
    PyTorchCallback
)

from configs.search_config import search_cfg
from configs.imagenet_train_cfg import cfg
from dataset import imagenet_data
from models import model_derived
from tools import utils
from tools.config_yaml import merge_cfg_from_file, update_cfg_from_cfg
from tools.lr_scheduler import get_lr_scheduler
from tools.multadds_count import comp_multadds

from .optimizer import Optimizer
from .trainer import SearchTrainer

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
        config = cfg

        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.cuda()

        SearchSpace = importlib.import_module('models.search_space_'+self.hparams.net_type).Network
        ArchGenerater = importlib.import_module('.derive_arch_'+self.hparams.net_type, __package__).ArchGenerate
        derivedNetwork = getattr(model_derived, '%s_Net' % self.hparams.net_type.upper())

        super_model = SearchSpace(self.hparams.optim.init_dim, self.hparams.data.dataset, config)
        arch_gener = ArchGenerater(super_model, config)
        der_Net = lambda net_config: derivedNetwork(net_config,
                                                    config=config)
        super_model = nn.DataParallel(super_model)
        super_model = super_model.cuda()

    if self.hparams.optim.sub_obj.type=='flops':
        flops_list, total_flops = super_model.module.get_cost_list(
                                config.data.input_size, cost_type='flops')
        super_model.module.sub_obj_list = flops_list
        print("Super Network flops (M) list: \n")
        print(str(flops_list))
        print("Total flops: " + str(total_flops))
    elif config.optim.sub_obj.type=='latency':
        with open(os.path.join('latency_list', config.optim.sub_obj.latency_list_path), 'r') as f:
            latency_list = eval(f.readline())
        super_model.module.sub_obj_list = latency_list
        print("Super Network latency (ms) list: \n")
        print(str(latency_list))
    else:
        raise NotImplementedError