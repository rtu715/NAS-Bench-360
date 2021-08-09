#!/usr/bin/env python

"""
Testing for ZeroShotNAS, aka data description driven nas
ZZ
May 14, 2020
"""

import tensorflow as tf
import numpy as np
import os
import logging
import pickle
import pandas as pd
import argparse

from amber.modeler import EnasCnnModelBuilder
from amber.architect.controller import ZeroShotController
from amber.architect.modelSpace import State, ModelSpace
from amber.architect.commonOps import count_model_params

from amber.architect.manager import EnasManager
from amber.architect.trainEnv import MultiManagerEnvironment
from amber.architect.reward import LossAucReward
from amber.plots import plot_controller_hidden_states
from amber.utils import run_from_ipython
from amber.utils.logging import setup_logger
from amber.utils.data_parser import get_data_from_simdata

from amber.modeler.modeler import build_sequential_model
from amber.architect.manager import GeneralManager
from amber.architect.modelSpace import get_layer_shortname

from amber.bootstrap.mock_manager import MockManager

import keras.backend as K
from keras.optimizers import SGD, Adam

from amber.bootstrap.gold_standard import get_gold_standard

manager_replica = 1

def get_controller(model_space, session, layer_embedding_sharing, data_description_len=3, is_enas=True, config_dict={}):
    with tf.device("/cpu:0"):
        controller = ZeroShotController(
            data_description_config={
                "length": data_description_len,
                "hidden_layer": {"units":config_dict.pop("descriptor_h", 8), "activation": "relu"},
                "regularizer": {"l1": config_dict.pop("descriptor_l1", 1e-8) }
                },
            model_space=model_space,
            session=session,
            share_embedding=layer_embedding_sharing,
            with_skip_connection=False,
            skip_weight=None,
            lstm_size=config_dict.pop("lstm_size", 32),
            lstm_num_layers=config_dict.pop("lstm_num_layers", 1),
            kl_threshold=config_dict.pop("kl_threshold", 0.05),
            train_pi_iter=100,
            optim_algo='adam',
            temperature=config_dict.pop("temperature", 1.5),
            tanh_constant=config_dict.pop("tanh_constant", 2),
            buffer_type="MultiManager",
            buffer_size=1 if is_enas else 5,
            batch_size=config_dict.pop("batch_size", 20),
            use_ppo_loss=config_dict.pop("use_ppo_loss", True),
            rescale_advantage_by_reward=False,
            verbose=False
        )
    return controller


def get_model_space_enas(out_filters=16, num_layers=4, num_pool=2):
    state_space = ModelSpace()
    expand_layers = [num_layers//num_pool*i-1 for i in range(num_pool)]
    assert num_layers%2 == 0
    layer_embedding_sharing = {}
    for i in range(0, num_layers, 2):
        state_space.add_layer(i, [
            State('conv1d', filters=out_filters, kernel_size=8, activation='relu'),
            State('conv1d', filters=out_filters, kernel_size=14, activation='relu'),
            State('conv1d', filters=out_filters, kernel_size=20, activation='relu'),
        ])
        state_space.add_layer(i+1, [
            State('maxpool1d', filters=out_filters, pool_size=8, strides=8),
            State('avgpool1d', filters=out_filters, pool_size=8, strides=8)
        ])
        if i in expand_layers:
            out_filters *= 2
        if i>0:
            layer_embedding_sharing[i] = 0
            layer_embedding_sharing[i+1] = 1
    return state_space, layer_embedding_sharing


def get_model_space_common():
    state_space = ModelSpace()
    state_space.add_layer(0, [
        State('conv1d', filters=3, kernel_size=8, kernel_initializer='glorot_uniform', activation='relu',
              name="conv1"),
        State('conv1d', filters=3, kernel_size=14, kernel_initializer='glorot_uniform', activation='relu',
              name="conv1"),
        State('conv1d', filters=3, kernel_size=20, kernel_initializer='glorot_uniform', activation='relu',
              name="conv1"),
    ])
    state_space.add_layer(1, [
        State('Identity'),
        State('maxpool1d', pool_size=8, strides=8),
        State('avgpool1d', pool_size=8, strides=8),

    ])
    state_space.add_layer(2, [
        State('Flatten'),
        State('GlobalMaxPool1D'),
        State('GlobalAvgPool1D'),
    ])
    state_space.add_layer(3, [
        State('Dense', units=3, activation='relu'),
        State('Dense', units=10, activation='relu'),
        State('Identity')
    ])
    return state_space, None



def read_data():
    dataset1 = get_data_from_simdata(
        positive_file="./data/zero_shot_sim/DensityEmbedding_prefix-MYC_known10_motifs-MYC_known10_min-1_max-10_mean-5_zeroProb-0_seqLength-1000_numSeqs-10000.simdata",
        negative_file="./data/zero_shot_sim/EmptyBackground_prefix-empty_bg_seqLength-1000_numSeqs-10000.simdata",
        targets=["MYC"])
    dataset2 = get_data_from_simdata(
        positive_file="./data/zero_shot_sim/DensityEmbedding_prefix-CTCF_known1_motifs-CTCF_known1_min-1_max-1_mean-1_zeroProb-0_seqLength-1000_numSeqs-10000.simdata",
        negative_file="./data/zero_shot_sim/EmptyBackground_prefix-empty_bg_seqLength-1000_numSeqs-10000.simdata",
        targets=["CTCF"])

    num_seqs = 20000
    train_idx = np.arange(0, int(num_seqs*0.8))
    val_idx = np.arange(int(num_seqs*0.8), int(num_seqs*0.9) )
    test_idx = np.arange(int(num_seqs*0.9), num_seqs )
    dictionarize = lambda dataset: {
            "train": (dataset[0][train_idx], dataset[1][train_idx]),
            "val": (dataset[0][val_idx], dataset[1][val_idx]),
            "test": (dataset[0][test_idx], dataset[1][test_idx]),
            }

    dataset1 = dictionarize(dataset1)
    dataset2 = dictionarize(dataset2)

    return dataset1, dataset2


def get_manager_enas(train_data, val_data, controller, model_space, wd, data_description, dag_name, verbose=2):
    input_node = State('input', shape=(1000, 4), name="input", dtype='float32')
    output_node = State('dense', units=1, activation='sigmoid')
    model_compile_dict = {
        'loss': 'binary_crossentropy',
        'optimizer': SGD(lr=0.01, momentum=0.9, decay=1e-5),
    }
    session = controller.session

    reward_fn = LossAucReward(method='auc')
    
    child_batch_size = 512
    model_fn = EnasCnnModelBuilder(
        dag_func='EnasConv1DwDataDescrption',
        batch_size=child_batch_size,
        session=session,
        model_space=model_space,
        inputs_op=[input_node],
        output_op=[output_node],
        num_layers=len(model_space),
        l1_reg=1e-8,
        l2_reg=5e-7,
        model_compile_dict=model_compile_dict,
        controller=controller,
        dag_kwargs={
            'with_skip_connection': False,
            'add_conv1_under_pool': False,
            'stem_config':{
                'has_stem_conv': False,
                'flatten_op': 'flatten',
                'fc_units': 30
                },
            'name': dag_name,
            'data_description': data_description
            }
    )
    
    manager = EnasManager(
        train_data=train_data,
        validation_data=val_data,
        epochs=1,
        child_batchsize=child_batch_size,
        reward_fn=reward_fn,
        model_fn=model_fn,
        store_fn='minimal',
        model_compile_dict=model_compile_dict,
        working_dir=wd,
        verbose=verbose
        )
    return manager


def get_manager_common(train_data, val_data, controller, model_space, wd, data_description, verbose=2, **kwargs):
    input_node = State('input', shape=(1000, 4), name="input", dtype='float32')
    output_node = State('dense', units=1, activation='sigmoid')
    model_compile_dict = {
        'loss': 'binary_crossentropy',
        'optimizer': 'adam',
        'metrics': ['acc']
    }

    reward_fn = LossAucReward(method='auc')

    child_batch_size = 500
    # TODO: convert functions in `_keras_modeler.py` to classes, and wrap up this Lambda function step
    model_fn = lambda model_arc: build_sequential_model(
        model_states=model_arc, input_state=input_node, output_state=output_node, model_compile_dict=model_compile_dict,
        model_space=model_space)
    manager = GeneralManager(
        train_data=train_data,
        validation_data=val_data,
        epochs=30,
        child_batchsize=child_batch_size,
        reward_fn=reward_fn,
        model_fn=model_fn,
        store_fn='model_plot',
        model_compile_dict=model_compile_dict,
        working_dir=wd,
        verbose=0,
        save_full_model=False,
        model_space=model_space
    )
    return manager


def get_manager_mock(dataset, model_space, **kwargs):
    if dataset in [1,2]:
        model_compile_dict = {
            'loss': 'binary_crossentropy',
            'optimizer': 'adam',
            'metrics': ['acc']
        }
        
        def mock_reward_fn(model_arc, train_history_df, *args, **kwargs):
            model_states_ = [str(model_space[i][model_arc[i]]) for i in range(len(model_arc))]
            idx_bool = np.array([train_history_df['L%i' % (i + 1)] == model_states_[i] for i in range(len(model_states_))])
            index = np.apply_along_axis(func1d=lambda x: all(x), axis=0, arr=idx_bool)
            l, k = train_history_df[['loss', 'knowledge']].iloc[np.random.choice(np.where(index)[0])]
            this_reward =  l 
            loss_and_metrics = [l, k]
            reward_metrics = {'knowledge': k}
            return this_reward, loss_and_metrics, reward_metrics

        manager = MockManager(
                #history_fn_list=['./outputs/zs_grid_history/dataset%i/zs_grid_%i/train_history.csv'%(dataset, i)
                #    for i in range(1,3)],
                history_fn_list = ['./outputs/zs_hist/dataset%i/train_history.csv'%dataset ],
                model_compile_dict=model_compile_dict,
                #metric_name_dict={"acc":0, "knowledge":1, "loss":2},
                metric_name_dict={"knowledge":0, "loss":1},
                reward_fn=mock_reward_fn
                )
    else:
        raise Exception("Unknown dataset: %s"%dataset)
    return manager


def get_bootstrap_gold_standard():
    model_space, _ = get_model_space_common()
    gs1, arch2id = get_gold_standard(['outputs/zs_hist/dataset1/train_history.csv'], model_space, metric_name_dict={'knowledge':0, 'loss':1}, id_remainder=81)
    gs2, _ = get_gold_standard(['outputs/zs_hist/dataset2/train_history.csv'], model_space, metric_name_dict={'knowledge':0, 'loss':1}, id_remainder=81)
    return gs1, gs2, arch2id


def get_samples_controller(dfeatures, controller, model_space, T=100, is_enas=True):
    probs = []
    actions = []
    for i in range(len(dfeatures)):
        dfeature = dfeatures[[i]]
        prob_arr = [ np.zeros((T, len(model_space[i]))) for i in range(len(model_space)) ]
        act_arr = []
        if is_enas:
            skip_layers = [i*2 for i in range(1, len(model_space))]
        else:
            skip_layers = []
        for t in range(T):
            a, p = controller.get_action(dfeature)
            act_arr.append(a)
            layer_id = 0
            for i, p in enumerate(p):
                if i in skip_layers:
                    continue
                prob_arr[layer_id][t] = p.flatten()
                layer_id += 1
        probs.append(prob_arr)
        actions.append(act_arr)
    return probs, actions


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


def reload(arg, controller=None):
    wd = arg.wd
    is_enas = arg.mode == 'enas'
    dfeatures = np.zeros((2*manager_replica, 2*manager_replica))
    np.fill_diagonal(dfeatures, 1)
    model_space, layer_embedding_sharing = get_model_space_common() if not is_enas else \
            get_model_space_enas(out_filters=4, num_layers=3, num_pool=3) 
    # build controller only if necessary
    if controller is None:
        try:
            session = tf.Session()
        except:
            session = tf.compat.v1.Session()
        K.set_session(session)

        controller = get_controller(model_space=model_space, session=session,
                data_description_len=dfeatures.shape[0],
                layer_embedding_sharing=layer_embedding_sharing,
                is_enas=is_enas)
    controller.load_weights(os.path.join(wd, "controller_weights.h5"))

    probs, actions = get_samples_controller(dfeatures, controller, model_space, T=1000, is_enas=is_enas)

    import seaborn as sns
    import matplotlib.pyplot as plt
    df = convert_to_dataframe(probs, model_space,
            data_names=['MYC_known10', 'CTCF_known1'] if manager_replica==1 else
                np.array([['MYC_%i'%i, 'CTCF_%i'%i] for i in range(manager_replica)]).flatten().tolist())

    for i in range(len(model_space)):
        sub_df = df.loc[ (df.layer==i) ]
        plt.clf()
        plt.tight_layout()
        ax = sns.boxplot(x="operation", y="prob",
            hue="description", palette=["m", "g"]*manager_replica,
            data=sub_df)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.savefig(os.path.join(wd, "layer_%i.png"%i), bbox_inches="tight")
    return probs, actions


def train(arg, config_dict=None, session=None, logger=None):
    wd = arg.wd
    config_dict = config_dict or {}
    is_enas = arg.mode == "enas"
    if logger is None:
        logger = setup_logger(wd, verbose_level=logging.INFO)
    verbose = 2
    model_space, layer_embedding_sharing = get_model_space_enas(out_filters=16, num_layers=4, num_pool=2) if is_enas else \
            get_model_space_common()
    #dfeatures = np.array([[1,0], [0,1]]*manager_replica)  # one-hot encoding
    dfeatures = np.zeros((2*manager_replica, 2*manager_replica))
    np.fill_diagonal(dfeatures, 1)

    if session is None:
        try:
            session = tf.Session()
        except:
            session = tf.compat.v1.Session()
        K.set_session(session)

    controller = get_controller(model_space=model_space, session=session, layer_embedding_sharing=layer_embedding_sharing,
            data_description_len=dfeatures.shape[0],
            is_enas=is_enas,
            config_dict=config_dict)

    if arg.mode == 'enas':
        dataset1, dataset2 = read_data()
        manager1 = get_manager_enas(train_data=dataset1['train'], val_data=dataset1['val'], controller=controller,
                model_space=model_space, wd=wd, data_description=dfeatures[[0]], dag_name="EnasDAG1", verbose=verbose)
        manager2 = get_manager_enas(train_data=dataset2['train'], val_data=dataset2['val'], controller=controller,
                model_space=model_space, wd=wd, data_description=dfeatures[[1]], dag_name="EnasDAG2", verbose=verbose)
        # only for enas
        vars_list1 = [v for v in tf.trainable_variables() if v.name.startswith(manager1.model_fn.dag.name)]
        vars_list2 = [v for v in tf.trainable_variables() if v.name.startswith(manager2.model_fn.dag.name)]
        # remove optimizer related vars (e.g. momentum, rms)
        vars_list1 = [v for v in vars_list1 if not v.name.startswith("%s/compile"%manager1.model_fn.dag.name)]
        vars_list2 = [v for v in vars_list2 if not v.name.startswith("%s/compile"%manager2.model_fn.dag.name)]
        logger.info("total model1 params: %i" % count_model_params(vars_list1))
        logger.info("total model2 params: %i" % count_model_params(vars_list2))
        with open(os.path.join(wd,"tensor_vars.txt"), "w") as f:
            for v in vars_list1 + vars_list2:
                f.write("%s\t%i\n"%(v.name, int(np.prod(v.shape).value) ))

    elif arg.mode == 'vanilla':
        dataset1, dataset2 = read_data()
        manager1 = get_manager_common(train_data=dataset1['train'], val_data=dataset1['val'], controller=controller,
                model_space=model_space, wd=wd, data_description=dfeatures[[0]], dag_name="EnasDAG1", verbose=verbose)
        manager2 = get_manager_common(train_data=dataset2['train'], val_data=dataset2['val'], controller=controller,
                model_space=model_space, wd=wd, data_description=dfeatures[[1]], dag_name="EnasDAG2", verbose=verbose)
    elif arg.mode == 'bootstrap':
        manager1 = get_manager_mock(dataset=1, model_space=model_space)
        manager2 = get_manager_mock(dataset=2, model_space=model_space)

    else:
        raise Exception("Cannot understand mode: %s"% arg.mode)


    env = MultiManagerEnvironment(
        data_descriptive_features= dfeatures,
        controller=controller,
        manager=[manager1, manager2]*manager_replica,
        logger=logger,
        max_episode=config_dict.pop("max_episode", 200),
        max_step_per_ep=config_dict.pop("max_step_per_ep", 15),
        working_dir=wd,
        time_budget="8:00:00",
        with_input_blocks=False,
        with_skip_connection=False,
        child_warm_up_epochs=2 if is_enas else 0
    )

    try:
        env.train()
    except KeyboardInterrupt:
        print("user interrupted training")
        pass
    controller.save_weights(os.path.join(wd, "controller_weights.h5"))
    return session, controller, [manager1, manager2]


def train_and_reload(arg):
    B = 50
    par_wd = arg.wd
    logger = setup_logger(par_wd, verbose_level=logging.CRITICAL)
    from zs_configs import get_zs_controller_configs
    configs_all = get_zs_controller_configs()
    gs1, gs2, arch2id = get_bootstrap_gold_standard()
    gs_list = [gs1, gs2]
    model_space, _ = get_model_space_common()
    # performance summary dataframe
    sum_df = pd.DataFrame(columns=['c', 'config_str', 'b', 'manager_index', 'target_mean', 'other_mean', 'target_median', 'other_median', 'target_sd', 'other_sd', 'target_rank_mean', 'other_rank_mean','target_rank_sd', 'other_rank_sd'])
    for c in range(len(configs_all)):
        configs = dict(configs_all[c])
        print(configs)
        for b in range(B):
            print('-'*20 + str(b) + '-'*20)
            arg.wd = os.path.join(par_wd, "config_%i"%c, "run_%i"%b)
            os.makedirs(arg.wd, exist_ok=True)
            graph = tf.Graph()
            session = tf.Session(graph=graph)
            with graph.as_default(), session.as_default():
                session, controller, manager_list = train(arg,
                        config_dict=configs,
                        session=session,
                        logger=logger)
                print("-"*20 + "finish training" + "-"*20)
                probs, actions = reload(arg, controller=controller)
                print("-"*20 + "finish reloading" + "-"*20)

            # get rewards for each manager
            for i in range(len(manager_list)):
                reward_arr = np.zeros((len(actions[i]), len(manager_list)))
                rank_arr = np.zeros((len(actions[i]), len(manager_list)))
                for j, a in enumerate(actions[i]):
                    for k, manager in enumerate(manager_list):
                        reward, loss_and_metrics = manager.get_rewards(-1, a)
                        reward_arr[j,k] = reward
                        this_arc = tuple([get_layer_shortname(model_space[l][a[l]]) for l in range(len(a))])
                        rank_arr[j,k] = gs_list[k].loc[gs_list[k].ID==arch2id[this_arc], 'loss_rank'].values[0]

                sum_df = sum_df.append(
                        {
                            'c': int(c),
                            'config_str': str(configs_all[c]),
                            'b': int(b),
                            'manager_index': int(i),
                            'target_mean': np.mean(reward_arr[:,i], keepdims=False),
                            'target_median': np.median(reward_arr[:,i], keepdims=False),
                            'target_sd': np.std(reward_arr[:,i], keepdims=False),
                            'other_mean': np.mean(reward_arr[:, [x for x in range(reward_arr.shape[1]) if x!=i]], keepdims=False),
                            'other_median': np.median(reward_arr[:, [x for x in range(reward_arr.shape[1]) if x!=i]], keepdims=False),
                            'other_sd': np.std(reward_arr[:, [x for x in range(reward_arr.shape[1]) if x!=i]], keepdims=False),
                            'target_rank_mean': np.mean(rank_arr[:,i], keepdims=False),
                            'target_rank_sd': np.std(rank_arr[:,i], keepdims=False),
                            'other_rank_mean': np.mean(rank_arr[:,[x for x in range(reward_arr.shape[1]) if x!=i]], keepdims=False),
                            'other_rank_sd': np.std(rank_arr[:,[x for x in range(reward_arr.shape[1]) if x!=i]], keepdims=False),
                        },
                        ignore_index=True)


    sum_df.to_csv(os.path.join(par_wd, "sum_df.tsv"), sep="\t", index=False, float_format="%.4f")




if __name__ == "__main__":
    if not run_from_ipython():
        parser = argparse.ArgumentParser(description="experimental zero-shot nas")
        parser.add_argument("--analysis", type=str, choices=['train', 'reload', 'both'], required=True, help="analysis type")
        parser.add_argument("--mode", type=str, choices=['enas', 'vanilla', 'bootstrap'], required=True, help="amber mode")
        parser.add_argument("--wd", type=str, default="./outputs/zero_shot/", help="working dir")

        arg = parser.parse_args()

        os.makedirs(arg.wd, exist_ok=True)
        if arg.analysis == "train":
            train(arg)
        elif arg.analysis == "reload":
            reload(arg)
        elif arg.analysis == "both":
            train_and_reload(arg)
        else:
            raise Exception("Unknown analysis type: %s"% arg.analysis)

