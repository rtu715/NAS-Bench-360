"""Utilities for motif objective functions
Migrated from `motif.py`
ZZ
2020.5.2
"""

import tensorflow as tf
import numpy as np
from collections import defaultdict
from .generalObjMath import multinomial_KL_divergence


def compare_motif_diff_size(P, Q):
    '''find maximum match between two metrics
    P and Q with different sizes
    '''
    best_d = float('inf')
    # by padding, Q is always wider than P
    P_half_len = int(np.ceil(P.shape[0] / 2.))
    Q_pad = np.concatenate(
        [
            np.ones((P_half_len, Q.shape[1])) / Q.shape[1],
            Q,
            np.ones((P_half_len, Q.shape[1])) / Q.shape[1]
        ], axis=0)

    # find best match of P in Q
    for i in range(0, Q_pad.shape[0] - P.shape[0] + 1):
        d = multinomial_KL_divergence(P, Q_pad[i:(i + len(P))])
        if d < best_d:
            best_d = d

    best_d /= float(P.shape[0])
    return best_d


def remove_dup_motif(motif_dict, threshold=0.05):
    tmp = {}
    for motif_name in motif_dict:
        this_motif = motif_dict[motif_name]
        best_d = 100
        for _, ref_motif in tmp.items():
            new_d = compare_motif_diff_size(this_motif, ref_motif)
            if new_d < best_d:
                best_d = new_d
        if best_d <= threshold:
            continue
        else:
            tmp[motif_name] = this_motif
    return tmp


def make_output_annot(label_annot, cat_list):
    # cat_list = ['TF', 'Pol', 'DNase', 'Histone']
    # split_vec = [np.where(label_annot.category==x)[0] for x in cat_list]
    split_vec = []
    for x in cat_list:
        if type(x) is str:
            split_vec.append(np.where(label_annot.category == x)[0])
        elif type(x) in (list, tuple):
            split_vec.append([i for i in range(label_annot.shape[0]) if label_annot.category[i] in x])
        else:
            raise Exception("Unknown category: %s" % x)
    output_annot = []
    for i in range(len(cat_list)):
        for j in range(len(split_vec[i])):
            k = split_vec[i][j]
            output_annot.append({
                'block': i,
                'index': j,
                'label_category': label_annot.iloc[k].category,
                'label_name': "%s_%s" % (label_annot.iloc[k].target, label_annot.iloc[k].cell),
                'label_index': k
            })
    return output_annot


def basewise_bits(motif):
    epsilon = 0.001
    assert motif.shape[1] == 4
    ent = np.apply_along_axis(lambda x: -np.sum(x * np.log2(np.clip(x, epsilon, 1 - epsilon))), 1, motif)
    return ent


def group_motif_by_factor(motif_dict, output_factors, threshold=0.1, pure_only=True):
    new_dict = defaultdict(dict)
    black_list_factors = ['CTCF', 'NFKB']
    for motif_name in motif_dict:
        factor = motif_name.split('_')[0]
        if not factor in output_factors:
            continue
        if factor in black_list_factors:
            continue
        if pure_only:
            ent = basewise_bits(motif_dict[motif_name])
            ENT_CUTOFF = 0.5
            MIN_ENT_BASES = 5
            if np.sum(ent < ENT_CUTOFF) < MIN_ENT_BASES:
                continue
            # if not motif_name.split('_')[1].startswith('known'):
            #    continue
            # cutoff = 0.98
            # if np.max(motif_dict[motif_name])<cutoff:
            #     continue
            # TODO: add trimming flanking to preserve only the consecutive 1s; trim flanking
            # rowmax = np.apply_along_axis(np.max, 1, motif_dict[motif_name])
            # one_loc = np.where(rowmax>=cutoff)[0]
            # if max(one_loc) - min(one_loc) < 4:
            #   continue
            # motif_dict[motif_name] = motif_dict[motif_name][min(one_loc):max(one_loc)+1]
        new_dict[factor].update({motif_name: motif_dict[motif_name]})
    for factor in new_dict:
        new_dict[factor] = remove_dup_motif(new_dict[factor], threshold=threshold)
    return new_dict


def get_seq_chunks(idx, min_cont=5, max_gap=3):
    intv = idx[1:] - idx[:-1]
    break_points = np.concatenate([[-1], np.where(intv > max_gap)[0], [idx.shape[0] - 1]])
    if len(break_points) <= 3:
        return []
    chunks = []
    for i in range(0, len(break_points) - 1):
        tmp = idx[[break_points[i] + 1, break_points[i + 1]]]
        if tmp[1] - tmp[0] > min_cont:
            chunks.append(tmp)
    return chunks


def convert_ppm_to_pwm(m, eps=1e-3):
    m = np.clip(m, eps, 1 - eps)
    # entropy = np.apply_along_axis(lambda p: -np.sum(p*np.log(p)), 1, m)
    # m_ = m/np.expand_dims(entropy, axis=-1)
    m_ = np.log2(m) - np.log2(0.25)
    return m_


def create_conv_from_motif(input_ph, motifs, factor_name='None'):
    max_motif_len = max([m.shape[0] for m in motifs.values()])
    new_motifs = []
    for m in motifs.values():
        left = (max_motif_len - m.shape[0]) // 2
        right = max_motif_len - m.shape[0] - left
        m_ = np.concatenate([np.ones((left, 4)) / 4., m, np.ones((right, 4)) / 4.])
        m_ = convert_ppm_to_pwm(m_)
        new_motifs.append(m_)
    w = np.stack(new_motifs).transpose([1, 2, 0])
    with tf.variable_scope(factor_name, reuse=tf.AUTO_REUSE):
        conv_w = tf.Variable(w, dtype=tf.float64, name='conv_w')
        conv_out = tf.nn.conv1d(input_ph, filters=conv_w, stride=1, padding="SAME", name='conv_out')
        conv_out = tf.reduce_max(conv_out, axis=-1)  # last dim is channel
    return conv_out, conv_w