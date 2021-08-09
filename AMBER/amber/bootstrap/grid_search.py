# -*- coding: UTF-8 -*-
# given a state space, perform B times exhaustive searches
# zzj, 2.23.2019

import csv
import itertools
import os

import numpy as np


def get_model_space_generator(model_space, with_skip_connection, with_input_blocks, num_input_blocks=1):
    new_space = []
    num_layers = len(model_space)
    for layer_id in range(num_layers):
        new_space.append([x for x in range(len(model_space[layer_id]))])
        if with_skip_connection:
            for i in range(layer_id):
                new_space.append([0, 1])
    if with_input_blocks:
        assert num_input_blocks > 1, "if `with_input_blocks=True`, expect `num_input_blocks>1`"
        ib_arr = combine_input_blocks(num_layers, num_input_blocks)
        insert_pos = np.zeros(num_layers, dtype='int32')
        insert_pos[0] = p = 1
        for i in range(1, num_layers):
            p += 1 + (i - 1) * with_skip_connection
            insert_pos[i] = p
        for n in itertools.product(*new_space):
            for ib in ib_arr:
                tmp = []
                # need a placeholder for input_blocks to concatenate
                # the last layer's skip connections
                for layer_op, layer_ib in zip(np.split(n, insert_pos), list(ib) + [np.array([])]):
                    tmp.extend(layer_op)
                    tmp.extend(layer_ib)
                yield np.array(tmp)
    else:
        return itertools.product(*new_space)


def combine_input_blocks(num_layers, num_input_blocks):
    """return all combinations of input_blocks when `input_block_unique_connection=True`
    """
    cmb_arr = np.zeros((num_layers ** num_input_blocks, num_layers, num_input_blocks), dtype='int32')
    cmb_list = [list(range(num_layers)) for _ in range(num_input_blocks)]
    idx_g = list(itertools.product(*cmb_list))
    for i in range(cmb_arr.shape[0]):
        idxs = idx_g[i]
        # idxs[j] is the "layer" which j-th input block is connected to
        for j in range(len(idxs)):
            cmb_arr[i, idxs[j], j] = 1
    return cmb_arr


def train_hist_csv_writter(writer, trial, loss_and_metrics, reward, model_states):
    data = [
        trial,
        [loss_and_metrics[x] for x in sorted(loss_and_metrics.keys())],
        reward
    ]
    action_list = [str(x) for x in model_states]
    data.extend(action_list)
    writer.writerow(data)
    print(action_list)


def rewrite_train_hist(working_dir, model_fn, knowledge_fn, data, suffix='new',
                       metric_name_dict={'acc': 0, 'knowledge': 1, 'loss': 2}):
    import tensorflow as tf
    from ..utils.io import read_history
    old_df = read_history([os.path.join(working_dir, "train_history.csv")], metric_name_dict)
    new_fh = open(os.path.join(working_dir, "train_history-%s.csv" % suffix), 'w')
    csv_writter = csv.writer(new_fh)
    total_layers = max([int(x.lstrip('L')) for x in old_df.columns.values if x.lstrip('L').isdigit()]) + 1
    for i in range(old_df.shape[0]):
        id = old_df['ID'][i]
        param_fp = os.path.join(old_df['dir'][i], 'weights', 'trial_%i' % id, 'bestmodel.h5')
        arc = np.array([old_df['L%i' % l][i] for l in range(1, total_layers)], dtype=np.int32)
        train_graph = tf.Graph()
        train_sess = tf.Session(graph=train_graph)
        with train_graph.as_default(), train_sess.as_default():
            model = model_fn(arc)
            model.load_weights(param_fp)
            new_k = knowledge_fn(model, data)
        loss_and_metrics = ({x: old_df[x][i] for x in metric_name_dict})
        loss_and_metrics.update({'knowledge': new_k})
        reward = old_df['loss'][i] + new_k
        train_hist_csv_writter(csv_writter, id, loss_and_metrics, reward, arc)
        new_fh.flush()
    new_fh.close()
    return


def grid_search(model_space_generator, manager, working_dir, B=10, resume_prev_run=True):
    write_mode = "a" if resume_prev_run else "w"
    fh = open(os.path.join(working_dir, 'train_history.csv'), write_mode)
    writer = csv.writer(fh)
    i = 0
    for b in range(B):
        # for backward comparability
        if getattr(model_space_generator, "__next__", None):
            model_space_generator_ = model_space_generator
        else:
            model_space_generator_ = itertools.product(*model_space_generator)
        for model_states in model_space_generator_:
            i += 1
            print("B={} i={} arc={}".format(b, i, ','.join([str(x) for x in model_states])))
            if not os.path.isdir(os.path.join(working_dir, 'weights', "trial_%i" % i)):
                os.makedirs(os.path.join(working_dir, 'weights', "trial_%i" % i))
            if resume_prev_run and os.path.isfile(os.path.join(working_dir, 'weights', "trial_%i" % i, "bestmodel.h5")):
                continue
            # reward, loss_and_metrics = manager.get_rewards(trial=i, model_states=model_states)
            reward, loss_and_metrics = manager.get_rewards(i, model_states)
            train_hist_csv_writter(writer, i, loss_and_metrics, reward, model_states)
            fh.flush()
    fh.close()
    return
