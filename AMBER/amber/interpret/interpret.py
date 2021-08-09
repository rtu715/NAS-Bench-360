# -*- coding: UTF-8 -*-

"""General functions for interpretig model weights
"""

import numpy as np
from ..modeler._operators import Layer_deNovo, SeparableFC, sparsek_vec
# custom layers
from keras.models import load_model

from .. import modeler as model_fn

custom_objects = {'SeparableFC': SeparableFC, 'Layer_deNovo': Layer_deNovo, 'sparsek_vec': sparsek_vec}


def load_full_model(modelPath):
    model = load_model(modelPath, custom_objects=custom_objects)
    return model


def get_models_from_hist_by_load(hist_idx, hist):
    model_dict = {}
    for idx in hist_idx:
        model_dict[idx] = load_full_model(
            '%s/weights/trial_%i/full_bestmodel.h5' % (hist.iloc[idx].dir, hist.iloc[idx].ID))
    return model_dict


def get_best_model(state_space, controller, working_directory):
    """Given a controller and a state_space, find the best model states in its
    state space

    Returns:
        dict: a dict of conditions for selected best model(s)
    """

    return


def get_hist_index_by_conditions(condition_dict, hist, complementary=False):
    """Get a set of indices for models that satisfy certain
    conditions from a hist file
    """
    sign_dict = {
        "==": lambda x, y: x == y,
        ">": lambda x, y: x > y,
        "<": lambda x, y: x < y,
        ">=": lambda x, y: x >= y,
        "<=": lambda x, y: x <= y,
        "!=": lambda x, y: x != y}
    condition_bool = []
    for cond in condition_dict:
        condition_bool.append(
            sign_dict[condition_dict[cond][0]](hist[cond], condition_dict[cond][1])
        )
    if complementary:
        subset = np.where(~np.all(np.array(condition_bool), axis=0))[0]
    else:
        subset = np.where(np.all(np.array(condition_bool), axis=0))[0]
    return subset


def get_models_from_hist(hist_idx, hist, input_state, output_state, state_space, model_compile_dict):
    """Given a set of indcies, build a dictionary of
    models from history file
    """
    model_dict = {}
    for idx in hist_idx:
        model_state_str = [hist.iloc[idx]["L%i" % (i + 1)] for i in range(hist.shape[1] - 5)]
        model_dict[idx] = model_fn.build_sequential_model_from_string(model_state_str, input_state, output_state,
                                                                      state_space, model_compile_dict)
        model_dict[idx].load_weights('%s/weights/trial_%i/bestmodel.h5' % (hist.iloc[idx].dir, hist.iloc[idx].ID))

    return model_dict


def get_multi_gpu_models_from_hist(hist_idx, hist, input_state, output_state, state_space, model_compile_dict):
    """Given a set of indcies, build a dictionary of
    models from history file
    """
    model_dict = {}
    for idx in hist_idx:
        model_state_str = [hist.iloc[idx]["L%i" % (i + 1)] for i in range(hist.shape[1] - 5)]
        model_dict[idx] = model_fn.build_multi_gpu_sequential_model_from_string(model_state_str, input_state,
                                                                                output_state, state_space,
                                                                                model_compile_dict)
        model_dict[idx].load_weights('%s/weights/trial_%i/bestmodel.h5' % (hist.iloc[idx].dir, hist.iloc[idx].ID))
    return model_dict


def match_quantity(q, p, num_slice=10, num_sample_per_slice=None, replace=False, random_state=777):
    """from p samples a new distribution that matches q

    Returns:
        np.array: sampled idx in p
    """
    np.random.seed(random_state)
    if not num_sample_per_slice:
        num_sample_per_slice = int(np.ceil(len(q) / (num_slice)))
    q_quantiles = np.percentile(q, q=np.arange(0, 100 + 100 / num_slice, 100 / num_slice))

    new_p_idx = []
    for q_s, q_e in zip(q_quantiles[:-1], q_quantiles[1:]):
        t = np.random.choice(
            np.where((p >= q_s) & (p <= q_e))[0],
            num_sample_per_slice,
            replace=replace)
        new_p_idx.extend(t)

    new_p_idx = np.array(new_p_idx)
    return new_p_idx
