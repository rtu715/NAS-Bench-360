import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from ..plots import sma
from ..architect.modelSpace import get_layer_shortname


def read_history_set(fn_list):
    tmp = []
    for fn in fn_list:
        t = pd.read_table(fn, sep=",", header=None)
        t['dir'] = os.path.dirname(fn)
        tmp.append(t)
    d = pd.concat(tmp)
    return d


def read_history(fn_list,
                 metric_name_dict={'acc': 0, 'knowledge': 1, 'loss': 2}):
    d = read_history_set(fn_list)
    d.columns = ['ID', 'metrics', 'reward'] + ['L%i' % i for i in range(1, d.shape[1] - 3)] + ['dir']
    metrics = {x: [] for x in metric_name_dict}
    for i in range(d.shape[0]):
        tmp = d.iloc[i, 1].split(',')
        for name, idx in metric_name_dict.items():
            t = tmp[idx]
            if idx == 0:
                t = t[1:]
            elif idx == len(tmp) - 1:
                t = t[:-1]
            t = t.strip("[]")
            t = float(t)
            metrics[name].append(t)
    for name in metrics:
        d[name] = metrics[name]
    d.drop(columns=['reward', 'metrics'], inplace=True)
    print("read %i history files, total %i lines " % (len(fn_list), d.shape[0]))
    return d


def read_action_weights(fn):
    """read 'weight_data.json' and derive the max likelihood
    architecture for each run. 'weight_data.json' stores the weight probability
    by function `save_action_weights` for a bunch of independent mock BioNAS
    optimization runs.
    """
    with open(fn, 'r') as f:
        data = json.load(f)
    archs = []
    tmp = list(data['L0']['operation'].keys())
    B = len(data['L0']['operation'][tmp[0]])
    for b in range(B):
        this_arch = []
        for l in range(len(data)):
            this_layer = {k: data["L%i" % l]['operation'][k][b][-1] for k in data["L%i" % l]['operation']}
            this_arch.append(
                max(this_layer, key=this_layer.get)
            )
        archs.append(tuple(this_arch))
    return archs


def read_action_weights_old(fn):
    """read 'weight_data.json' and derive the max likelihood
    architecture for each run. 'weight_data.json' stores the weight probability
    by function `save_action_weights` for a bunch of independent mock BioNAS
    optimization runs.
    """
    with open(fn, 'r') as f:
        data = json.load(f)
    archs = []
    tmp = list(data['L0'].keys())
    B = len(data['L0'][tmp[0]])
    for b in range(B):
        this_arch = []
        for l in range(len(data)):
            this_layer = {k: data["L%i" % l][k][b][-1] for k in data["L%i" % l]}
            this_arch.append(
                max(this_layer, key=this_layer.get)
            )
        archs.append(tuple(this_arch))
    return archs


def annotate_probs_list(probs_list, model_space, with_input_blocks, with_skip_connection):
    """for a given probs_list, annotate what is each prob about

    Parameters
    ----------
        probs_list:
        model_space:
        with_skip_connection
        with_input_blocks
    """
    #	data_per_layer: a list of length of total model architecture sequence, where each slice is
    #		a list of length `GeneralController.batch_size`
    data_per_layer = list(zip(*probs_list))
    ops_pointer = 0
    num_layers = len(model_space)
    data_dict = defaultdict(dict)
    for layer_id in range(num_layers):
        data_dict['L%i' % layer_id]['operation'] = data_per_layer[ops_pointer]
        if with_input_blocks:
            data_dict['L%i' % layer_id]['input_blocks'] = data_per_layer[ops_pointer + 1]
        if with_skip_connection:
            if layer_id > 0:
                data_dict['L%i' % layer_id]['skip_connection'] = data_per_layer[
                    ops_pointer + 1 + int(with_input_blocks)]
            else:
                data_dict['L%i' % layer_id]['skip_connection'] = []
        ops_pointer += 1 + 1 * with_input_blocks + 1 * with_skip_connection * int(layer_id > 0)
    return data_dict


def save_action_weights(probs_list, state_space, working_dir, with_input_blocks=False, with_skip_connection=False,
                        **kwargs):
    """
    Parameters
    ----------
    probs_list: list
        list of probability at each time step output a series of graphs each plotting weight of options of
        each layer over time

    Note
    --------
        if `with_input_blocks` is True, then expect `input_nodes` in keyword_arguments
        `input_nodes` is a List of BioNAS.Controller.state_space.State, hence the layer name
        can be accessed by State.Layer_attributes['name']
    """
    assert not (with_input_blocks ^ (
                'input_nodes' in kwargs)), "if `with_input_blocks` is True, must provide `input_nodes` " \
                                           "in keyword_arguments "
    data_dict = annotate_probs_list(probs_list, state_space,
                                    with_input_blocks=with_input_blocks,
                                    with_skip_connection=with_skip_connection)
    save_path = os.path.join(working_dir, 'weight_data.json')
    if not os.path.exists(save_path):
        df = {}
        for layer, state_list in enumerate(state_space):
            df['L%i' % layer] = defaultdict(dict)
            for k in state_list:
                t = get_layer_shortname(k)
                df['L' + str(layer)]['operation'][t] = []
            if with_input_blocks:
                input_block_names = [n.Layer_attributes['name'] for n in kwargs['input_nodes']]
                for n in input_block_names:
                    df['L%i' % layer]['input_blocks'][n] = []
            if with_skip_connection and layer > 0:
                df['L%i' % layer]['skip_connection'] = {}
                for i in range(layer):
                    df['L%i' % layer]['skip_connection']['from_L%i' % i] = []

    else:
        with open(save_path, 'r+') as f:
            df = json.load(f)

    for layer, state_list in enumerate(state_space):
        try:
            data = data_dict['L%i' % layer]['operation']
        except KeyError:
            print(layer, data_dict['L%i' % layer].keys())
            raise Exception('above')
        data = [p.squeeze().tolist() for p in data]
        data_per_type = list(zip(*data))
        total_len = len(data_per_type[0])
        for i, d in enumerate(data_per_type):
            k = state_list[i]
            t = get_layer_shortname(k)
            df['L' + str(layer)]['operation'][t].append(sma(d).tolist())
        if with_input_blocks:
            input_block_names = [n.Layer_attributes['name'] for n in kwargs['input_nodes']]
            ib_data = np.array(data_dict['L%i' % layer]['input_blocks'])[:, :, :, 1].squeeze().transpose()
            for j in range(len(input_block_names)):
                df['L' + str(layer)]['input_blocks'][input_block_names[j]].append(sma(ib_data[j, :]).tolist())
        if with_skip_connection and layer > 0:
            sc_data = np.array(data_dict['L%i' % layer]['skip_connection'])[:, :, :, 1].squeeze().transpose()
            sc_data = sc_data.reshape((layer, total_len))
            for i in range(layer):
                df['L' + str(layer)]['skip_connection']['from_L%i' % i].append(sma(sc_data[i, :]).tolist())

    with open(save_path, 'w') as f:
        json.dump(df, f)


def save_stats(loss_and_metrics_list, working_dir):
    save_path = os.path.join(working_dir, 'nas_training_stats.json')
    if not os.path.exists(save_path):
        df = {'Knowledge': [], 'Accuracy': [], 'Loss': []}
    else:
        with open(save_path, 'r+') as f:
            df = json.load(f)

    keys = list(loss_and_metrics_list[0].keys())
    data = [list(loss_and_metrics.values()) for loss_and_metrics in loss_and_metrics_list]
    data_per_cat = list(zip(*data))
    k_data = data_per_cat[keys.index('knowledge')]
    loss_data = data_per_cat[keys.index('loss')]
    df['Knowledge'].append(list(k_data))
    df['Loss'].append(list(loss_data))
    # modified 2020.6.7 by ZZ: not every case will have the Accuracy metric..
    try:
        acc_data = data_per_cat[keys.index('acc')]
        df['Accuracy'].append(list(acc_data))
    except ValueError:
        pass
    with open(save_path, 'w') as f:
        json.dump(df, f)
