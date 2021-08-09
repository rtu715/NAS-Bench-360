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
import sys
import copy
import logging
import pickle
import pandas as pd
import argparse

from amber.modeler import EnasCnnModelBuilder
from amber.architect.controller import ZeroShotController
from amber.architect.modelSpace import State, ModelSpace
from amber.architect.commonOps import count_model_params, unpack_data

from amber.architect.trainEnv import MultiManagerEnvironment, ParallelMultiManagerEnvironment
from amber.architect.reward import LossAucReward, LossReward
from amber.plots import plot_controller_hidden_states
from amber.utils import run_from_ipython, get_available_gpus
from amber.utils.logging import setup_logger
from amber.utils.data_parser import get_data_from_simdata

from amber.modeler.modeler import build_sequential_model, KerasModelBuilder
from amber.architect.manager import GeneralManager, DistributedGeneralManager
from amber.architect.modelSpace import get_layer_shortname

from amber.utils.sequences import EncodedGenome
from amber.utils.sequences import EncodedHDF5Genome
from amber.utils.sampler import BatchedHDF5Generator


def get_controller(model_space, session, data_description_len=3, layer_embedding_sharing=None):
    with tf.device("/cpu:0"):
        controller = ZeroShotController(
            data_description_config={
                "length": data_description_len,
                #"hidden_layer": {"units":8, "activation": "relu"},
                #"regularizer": {"l1":1e-8}
                },
            share_embedding=layer_embedding_sharing,
            model_space=model_space,
            session=session,
            with_skip_connection=False,
            skip_weight=None,
            skip_target=0.2,
            lstm_size=128,
            lstm_num_layers=2,
            kl_threshold=0.05,
            train_pi_iter=100,
            optim_algo='adam',
            temperature=0.5,
            tanh_constant=2,
            buffer_type="MultiManager",
            buffer_size=5,
            batch_size=20,
            use_ppo_loss=False,
            rescale_advantage_by_reward=False
        )
    return controller


def get_model_space_common():

    # Setup and params.
    state_space = ModelSpace()
    default_params = {"kernel_initializer": "glorot_uniform",
                      "activation": "relu"}
    param_list = [
            # Block 1:
            #[
            #    {"filters": 64, "kernel_size": 8},
            #    {"filters": 64, "kernel_size": 14},
            #    {"filters": 64, "kernel_size": 20}
            #],
            # Block 2:
            #[
            #    {"filters": 128, "kernel_size": 8},
            #    {"filters": 128, "kernel_size": 14},
            #    {"filters": 128, "kernel_size": 20}
            #],
            # Block 3:
            [
                {"filters": 256, "kernel_size": 8},
                {"filters": 256, "kernel_size": 14},
                {"filters": 256, "kernel_size": 20}
            ],
        ]

    # Build state space.
    layer_embedding_sharing = {}
    conv_seen = 0
    for i in range(len(param_list)):
        # Build conv states for this layer.
        conv_states = [State("Identity")]
        for j in range(len(param_list[i])):
            d = copy.deepcopy(default_params)
            for k, v in param_list[i][j].items():
                d[k] = v
            conv_states.append(State('conv1d', name="conv{}".format(conv_seen), **d))
        state_space.add_layer(conv_seen*3, conv_states)
        if i > 0:
            layer_embedding_sharing[conv_seen*3] = 0
        conv_seen += 1

        # Add pooling states.
        if i < len(param_list) - 1:
            pool_states = [State('Identity'),
                           State('maxpool1d', pool_size=4, strides=4),
                           State('avgpool1d', pool_size=4, strides=4)]
            if i > 0:
                layer_embedding_sharing[conv_seen*3-2] = 1
        else:
            pool_states = [
                    State('Flatten'),
                    State('GlobalMaxPool1D'),
                    State('GlobalAvgPool1D')
                ]
        state_space.add_layer(conv_seen*3-2, pool_states)


        # Add dropout
        state_space.add_layer(conv_seen*3-1, [
            State('Identity'),
            State('Dropout', rate=0.1),
            State('Dropout', rate=0.3),
            State('Dropout', rate=0.5)
            ])
        if i > 0:
            layer_embedding_sharing[conv_seen*3-1] = 2

    # Add final classifier layer.
    state_space.add_layer(conv_seen*3, [
            State('Dense', units=30, activation='relu'),
            State('Dense', units=100, activation='relu'),
            State('Identity')
        ])
    return state_space, layer_embedding_sharing


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
        epochs=50,
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
            'steps_per_epoch': 100,
            'workers': 3, 'max_queue_size': 50,
            'earlystop_patience': 5}
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


def get_samples_controller(dfeatures, controller, model_space, T=100):
    res = []
    for i in range(len(dfeatures)):
        dfeature = dfeatures[[i]]
        prob_arr = [ np.zeros((T, len(model_space[i]))) for i in range(len(model_space)) ]
        for t in range(T):
            a, p = controller.get_action(dfeature)
            for i, p in enumerate(p):
                prob_arr[i][t] = p.flatten()
        res.append(prob_arr)
    return res


def convert_to_dataframe(res, model_space, data_names):
    probs = []
    layer = []
    description = []
    operation = []
    for i in range(len(res)):
        for j in range(len(model_space)):
            for k in range(len(model_space[j])):
                o = get_layer_shortname(model_space[j][k])
                p = res[i][j][:, k]
                # extend the array
                T = p.shape[0]
                probs.extend(p)
                description.extend([data_names[i]]*T)
                layer.extend([j]*T)
                operation.extend([o]*T)
    df = pd.DataFrame({
        'description':description,
        'layer': layer,
        'operation': operation,
        'prob': probs
        })
    return df


def reload_trained_controller(arg):
    wd = arg.wd #wd = "./outputs/zero_shot/"
    configs, config_keys, controller, model_space = read_configs(arg)
    controller.load_weights(os.path.join(wd, "controller_weights.h5"))
    dfeatures = np.stack([configs[k]["dfeatures"] for k in config_keys])
    res = get_samples_controller(dfeatures, controller, model_space, T=1000)
   
    import seaborn as sns
    import matplotlib.pyplot as plt
    df = convert_to_dataframe(res, model_space, data_names=[configs[k]['feat_name'] for k in config_keys])

    for i in range(len(model_space)):
        sub_df = df.loc[ (df.layer==i) ]
        plt.clf()
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))
        sns.boxplot(x="operation", y="prob",
            hue="description",
            data=sub_df,
            ax=ax
            )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(wd, "layer_%i.png"%i), bbox_inches="tight")
    df.to_csv(os.path.join(wd, "sampled_controller_arcs.tsv"), sep="\t")
    return res


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

    manager_getter = get_manager_distributed if arg.parallel else get_manager_common
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
                        'y_selector': configs[k]['ds_col'],
                        'batch_size': 1024 if arg.parallel else 1024*len(gpus),
                        'shuffle': x=='train',
                }
            if arg.parallel is True:
                configs[k][x] = BatchedHDF5Generator
                configs[k]['%s_data_kwargs'%x] = d
            else:
                configs[k][x] = BatchedHDF5Generator(**d)

        # Build covariates and manager.
        configs[k]["dfeatures"] = np.array(
            [configs[k][x] for x in dfeature_names]) # TODO: Make cols dynamic.

        tmp = dict() if arg.parallel is False else dict(
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
            verbose=0 if arg.parallel is True else 2,
            n_feats=configs[k]["n_feats"],
            **tmp
            )
        config_keys.append(k)
    return configs, config_keys, controller, model_space



def train_nas(arg):
    wd=arg.wd
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
               time_budget="72:00:00",
               with_input_blocks=False,
               with_skip_connection=False,
               save_controller_every=1,
               resume_prev_run=arg.resume
           )

    if arg.parallel is True:
        env = ParallelMultiManagerEnvironment(
                    processes=len(gpus),
                    **tmp)
    else:
        env = MultiManagerEnvironment(**tmp)

    try:
        env.train()
    except KeyboardInterrupt:
        print("user interrupted training")
        pass
    controller.save_weights(os.path.join(wd, "controller_weights.h5"))


def get_parser():
    parser = argparse.ArgumentParser(description="experimental zero-shot nas")
    parser.add_argument("--analysis", type=str, choices=['train', 'reload'], required=True, help="analysis type")
    parser.add_argument("--wd", type=str, default="./outputs/zero_shot/", help="working dir")
    parser.add_argument("--parallel", default=False, action="store_true", help="Use parallel")
    parser.add_argument("--resume", default=False, action="store_true", help="resume previous run")
    parser.add_argument("--config-file", type=str, required=True, help="Path to the config file to use.")
    parser.add_argument("--dfeature-name-file", type=str, required=True, help="Path to file with dataset feature names listed one per line.")
    parser.add_argument("--train-file", type=str, required=True, help="Path to the hdf5 file of training data.")
    parser.add_argument("--val-file", type=str, required=True, help="Path to the hdf5 file of validation data.")
    parser.add_argument("--lockstep-sampling", default=False, action="store_true", help="Ensure same training samples used for all models.")
    return parser


if __name__ == "__main__":
    if run_from_ipython():
        arg_str = """--analysis reload     --wd  ./outputs/zero_shot_deepsea/     --config-file ./data/zero_shot_deepsea/debug_feats.config.2_cats.tsv     --train-file ./data/zero_shot_deepsea/train.h5     --val-file ./data/zero_shot_deepsea/val.h5     --dfeature-name-file ./data/zero_shot_deepsea/dfeatures_ordered_list.txt     --parallel"""
        arg_list = arg_str.split()
        sys.argv = ['foo'] + arg_list
    else:
        arg = get_parser()
        arg = parser.parse_args()

        if arg.analysis == "train":
            train_nas(arg)
        elif arg.analysis == "reload":
            reload_trained_controller(arg)
        else:
            raise Exception("Unknown analysis type: %s"% arg.analysis)
