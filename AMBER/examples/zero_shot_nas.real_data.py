#!/usr/bin/env python

"""
Testing for ZeroShotNAS, aka data description driven nas
ZZ
May 14, 2020
"""


import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import os
import copy
import logging
import pickle
import pandas as pd
import argparse

from amber.architect.controller import ZeroShotController
from amber.architect.modelSpace import State, ModelSpace
from amber.architect.commonOps import count_model_params, unpack_data

from amber.architect.trainEnv import MultiManagerEnvironment, ParallelMultiManagerEnvironment
from amber.architect.reward import LossAucReward, LossReward
from amber.utils import run_from_ipython, get_available_gpus
from amber.utils.logging import setup_logger

from amber.architect.manager import GeneralManager, DistributedGeneralManager
from amber.architect.model_space import get_layer_shortname
from amber.modeler import KerasModelBuilder

from amber.utils.sampler import BatchedHDF5Generator, Selector

# put model space in zs_config.py
from zs_config import get_model_space_common

def get_controller(model_space, session, data_description_len=3, layer_embedding_sharing=None):
    with tf.device("/cpu:0"):
        controller = ZeroShotController(
            data_description_config={
                "length": data_description_len,
                #"hidden_layer": {"units":16, "activation": "relu"},
                #"regularizer": {"l1":1e-8 }
                },
            share_embedding=layer_embedding_sharing,
            model_space=model_space,
            session=session,
            with_skip_connection=False,
            skip_weight=None,
            lstm_size=64,
            lstm_num_layers=1,
            kl_threshold=0.01,
            train_pi_iter=100,
            optim_algo='adam',
            temperature=1.,
            tanh_constant=1.5,
            buffer_type="MultiManager",
            buffer_size=10,
            batch_size=5,
            use_ppo_loss=False,
            rescale_advantage_by_reward=True
        )
    return controller


def get_manager_distributed(train_data, val_data, controller, model_space, wd, data_description, verbose=0,
                            devices=None, train_data_kwargs=None, validate_data_kwargs=None, **kwargs):
    reward_fn = LossAucReward(method='auc')
    input_node = State('input', shape=(1000, 4), name="input", dtype='float32')
    output_node = State('dense', units=1, activation='sigmoid')
    model_compile_dict = {
        'loss': 'binary_crossentropy',
        'optimizer': 'adam',
        'metrics': ['acc']
    }
    mb = KerasModelBuilder(inputs=input_node, outputs=output_node, model_compile_dict=model_compile_dict, model_space=model_space)
    manager = DistributedGeneralManager(
        devices=devices,
        train_data_kwargs=train_data_kwargs,
        train_data=train_data,
        validate_data_kwargs=validate_data_kwargs,
        validation_data=val_data,
        epochs=30,
        child_batchsize=1000,
        reward_fn=reward_fn,
        model_fn=mb,
        store_fn='model_plot',
        model_compile_dict=model_compile_dict,
        working_dir=wd,
        verbose=verbose,
        save_full_model=False,
        model_space=model_space,
        fit_kwargs={
            'steps_per_epoch': 50,
            'workers': 3, 'max_queue_size': 50,
            'earlystop_patience': 10}
    )
    return manager


def get_manager_common(train_data, val_data, controller, model_space, wd, data_description, verbose=2, n_feats=1, **kwargs):
    input_node = State('input', shape=(1000, 4), name="input", dtype='float32')
    output_node = State('dense', units=n_feats, activation='sigmoid')
    model_compile_dict = {
        'loss': 'binary_crossentropy',
        'optimizer': 'adam',
        'metrics': ['acc']
    }
    session = controller.session

    reward_fn = LossAucReward(method='auc')
    gpus = get_available_gpus()
    num_gpus = len(gpus)
    mb = KerasModelBuilder(inputs=input_node, outputs=output_node, model_compile_dict=model_compile_dict, model_space=model_space, gpus=num_gpus)

    # TODO: batch_size here is not effective because it's been set at generator init
    child_batch_size = 1000*num_gpus
    manager = GeneralManager(
        train_data=train_data,
        validation_data=val_data,
        epochs=1000,
        child_batchsize=child_batch_size,
        reward_fn=reward_fn,
        model_fn=mb,
        store_fn='model_plot',
        model_compile_dict=model_compile_dict,
        working_dir=wd,
        verbose=verbose,
        save_full_model=True,
        model_space=model_space,
        fit_kwargs={
            'steps_per_epoch': 50,
            'workers': 8, 'max_queue_size': 50,
            'earlystop_patience': 20}
    )
    return manager


def read_configs(arg):
    dfeature_names = list()
    with open(arg.dfeature_name_file, "r") as read_file:
        for line in read_file:
            line = line.strip()
            if line:
                dfeature_names.append(line)
    wd = arg.wd
    model_space, layer_embedding_sharing = get_model_space_common()
    print(layer_embedding_sharing)
    try:
        session = tf.Session()
    except AttributeError:
        session = tf.compat.v1.Session()

    controller = get_controller(
            model_space=model_space,
            session=session,
            data_description_len=len(dfeature_names),
            layer_embedding_sharing=layer_embedding_sharing)
    # Load in datasets and configurations for them.
    if arg.config_file.endswith("tsv"):
        sep = "\t"
    else:
        sep = ","
    configs = pd.read_csv(arg.config_file, sep=sep)
    tmp = list(configs.columns) # Because pandas doesn't have infer quotes...
    if any(["\"" in x for x in tmp]):
        configs = pd.read_csv(arg.config_file, sep=sep, quoting=2)
        print("Re-read with quoting")
    configs = configs.to_dict(orient='index')
    # Get available gpus for parsing to DistributedManager
    gpus = get_available_gpus()
    if len(gpus) == 0:
        gpus = [None]
    gpus_ = gpus * len(configs)

    manager_getter = get_manager_distributed
    config_keys = list()
    seed_generator = np.random.RandomState(seed=1337)
    for i, k in enumerate(configs.keys()):
        # Build datasets for train/test/validate splits.
        for x in ["train", "validate"]:
            if arg.lockstep_sampling is False and x == "train":
                cur_seed = seed_generator.randint(0, np.iinfo(np.uint32).max)
            else:
                cur_seed = 1337
            d = {
                        'hdf5_fp':  arg.train_file if x=='train' else arg.val_file,
                        'x_selector': Selector(label='x'),
                        'y_selector': Selector(label='labels/%s'%configs[k]["feat_name"]),
                        'batch_size': 1024,
                        'shuffle': x=='train',
                }
            configs[k][x] = BatchedHDF5Generator
            configs[k]['%s_data_kwargs'%x] = d

        # Build covariates and manager.
        configs[k]["dfeatures"] = np.array(
            [configs[k][x] for x in dfeature_names]) # TODO: Make cols dynamic.

        tmp = dict(
                train_data_kwargs=configs[k]['train_data_kwargs'],
                validate_data_kwargs=configs[k]["validate_data_kwargs"]
                )

        configs[k]["manager"] = manager_getter(
            devices=[gpus_[i]],
            train_data=configs[k]["train"],
            val_data=configs[k]["validate"],
            controller=controller,
            model_space=model_space,
            wd=os.path.join(wd, "manager_%s"%k),
            data_description=configs[k]["dfeatures"],
            dag_name="AmberDAG{}".format(k),
            verbose=0,
            n_feats=configs[k]["n_feats"],
            **tmp
            )
        config_keys.append(k)
    return configs, config_keys, controller, model_space


def train_nas(arg):
    wd = arg.wd
    logger = setup_logger(wd, verbose_level=logging.INFO)

    gpus = get_available_gpus()
    configs, config_keys, controller, model_space = read_configs(arg)

    # Setup env kwargs.
    tmp = dict(data_descriptive_features=np.stack([configs[k]["dfeatures"] for k in config_keys]),
               controller=controller,
               manager=[configs[k]["manager"] for k in config_keys],
               logger=logger,
               max_episode=200,
               max_step_per_ep=5,
               working_dir=wd,
               time_budget="150:00:00",
               with_input_blocks=False,
               with_skip_connection=False,
               save_controller_every=1,
               enable_manager_sampling=True
           )

    env = ParallelMultiManagerEnvironment(
                processes=len(gpus) if len(gpus)>1 else 1,
                **tmp)

    try:
        env.train()
    except KeyboardInterrupt:
        print("user interrupted training")
        pass
    controller.save_weights(os.path.join(wd, "controller_weights.h5"))


if __name__ == "__main__":
    if not run_from_ipython():
        parser = argparse.ArgumentParser(description="experimental zero-shot nas")
        parser.add_argument("--train-file", type=str, required=True, help="Path to the hdf5 file of training data.")
        parser.add_argument("--val-file", type=str, required=True, help="Path to the hdf5 file of validation data.")
        parser.add_argument("--wd", type=str, default="./outputs/zero_shot/", help="working dir")
        parser.add_argument("--resume", default=False, action="store_true", help="resume previous run")
        parser.add_argument("--config-file", type=str, required=True, help="Path to the config file to use.")
        parser.add_argument("--dfeature-name-file", type=str, required=True, help="Path to file with dataset feature names listed one per line.")
        parser.add_argument("--lockstep-sampling", default=False, action="store_true", help="Ensure same training samples used for all models.")

        arg = parser.parse_args()

        train_nas(arg)
