#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example wrapper `Amber` use for searching single-cell expression prediction
FZZ, June 19, 2020
"""

from amber import Amber
from amber.utils.sampler import BatchedHDF5Generator, Selector
from amber.utils import run_from_ipython, get_available_gpus
from amber.architect import ModelSpace, Operation
import sys
import os
import pickle
import numpy as np
import scipy.stats as ss
#from tensorflow.keras.optimizers import SGD, Adam
from keras.optimizers import SGD, Adam
import argparse
from zs_config import get_model_space_long_and_dilation as get_model_space
from zs_config import read_metadata


def get_data_config_amber_encoded(fp, feat_name, batch_size, shuffle):
    """Prepare the kwargs for BatchedHDF5Generator

    Note
    ------
    Only works for the layout of `data/zero_shot/amber_encoded.train_feats.train.h5`
    """
    d = {
            'hdf5_fp': fp,
            'x_selector': Selector('x'),
            'y_selector': Selector('labels/%s'%feat_name),
            'batch_size': batch_size,
            'shuffle': shuffle
        }
    return d


def get_data_config_deepsea_compiled(fp, feat_name, batch_size, shuffle):
    """Equivalent for amber encoded but for deepsea 919 compiled hdf5
    """
    meta = read_metadata()
    d = {
            'hdf5_fp': fp,
            'x_selector': Selector(label='x'),
            'y_selector': Selector(label='y', index=meta.loc[feat_name].col_idx),
            'batch_size': batch_size,
            'shuffle': shuffle
        }
    return d


def amber_app(wd, feat_name, run=False):
    # First, define the components we need to use
    type_dict = {
        'controller_type': 'GeneralController',
        'knowledge_fn_type': 'zero',
        'reward_fn_type': 'LossAucReward',

        # FOR RL-NAS
        'modeler_type': 'KerasModelBuilder',
        'manager_type': 'DistributedManager',
        'env_type': 'ControllerTrainEnv'
    }


    # Next, define the specifics
    train_data_kwargs = get_data_config_deepsea_compiled(
            #fp="./data/zero_shot/amber_encoded.train_feats.train.h5",
            fp="./data/zero_shot_deepsea/train.h5",
            feat_name=feat_name,
            batch_size=1024,
            shuffle=True
            )
    validate_data_kwargs = get_data_config_deepsea_compiled(
            #fp="./data/zero_shot/amber_encoded.train_feats.validate.h5",
            fp="./data/zero_shot_deepsea/val.h5",
            feat_name=feat_name,
            batch_size=1024,
            shuffle=False
            )
    os.makedirs(wd, exist_ok=True)

    input_node = [
            Operation('input', shape=(1000,4), name="input")
            ]

    output_node = [
            Operation('dense', units=1, activation='sigmoid', name="output")
            ]

    model_compile_dict = {
        'loss': 'binary_crossentropy',
        'optimizer': 'adam',
        #'optimizer': SGD(lr=1e-4, momentum=0.9),
        'metrics': ['acc']
    }

    model_space, layer_embedding_sharing = get_model_space()
    batch_size = 1024
    use_ppo = False

    specs = {
        'model_space': model_space,

        'controller': {
                'share_embedding': layer_embedding_sharing,
                'with_skip_connection': False,
                'skip_weight': None,
                'lstm_size': 128,
                'lstm_num_layers': 1,
                'kl_threshold': 0.1,
                'train_pi_iter': 100,
                'optim_algo': 'adam',
                'rescale_advantage_by_reward': False,
                'temperature': 1.0,
                'tanh_constant': 1.5,
                'buffer_size': 10,  # FOR RL-NAS
                'batch_size': 5,
                'use_ppo_loss': use_ppo
        },

        'model_builder': {
            'batch_size': batch_size,
            'inputs_op': input_node,
            'outputs_op': output_node,
            'model_compile_dict': model_compile_dict,
        },

        'knowledge_fn': {'data': None, 'params': {}},

        'reward_fn': {'method': "auc"},

        'manager': {
            'data': {
                'train_data': BatchedHDF5Generator,
                'validation_data': BatchedHDF5Generator,
            },
            'params': {
                'train_data_kwargs': train_data_kwargs,
                'validate_data_kwargs': validate_data_kwargs,
                'devices': ['/device:GPU:0'],
                'epochs': 100,
                'fit_kwargs': {
                    'earlystop_patience': 40,
                    'steps_per_epoch': 100,
                    'max_queue_size': 50,
                    'workers': 3
                    },
                'child_batchsize': batch_size,
                'store_fn': 'model_plot',
                'working_dir': wd,
                'verbose': 0
            }
        },

        'train_env': {
            'max_episode': 75,
            'max_step_per_ep': 5,
            'working_dir': wd,
            'time_budget': "24:00:00",
            'with_skip_connection': False,
            'save_controller_every': 1
        }
    }


    # finally, run program
    amb = Amber(types=type_dict, specs=specs)
    if run:
        amb.run()
    return amb


if __name__ == '__main__':
    if not run_from_ipython():
        parser = argparse.ArgumentParser(description="Script for AMBER-search of Single-task runner")
        parser.add_argument("--wd", type=str, help="working directory")
        parser.add_argument("--feat-name", type=str, help="feature name")

        args = parser.parse_args()

        amber_app(
                wd=args.wd,
                feat_name=args.feat_name,
                run=True
                )
