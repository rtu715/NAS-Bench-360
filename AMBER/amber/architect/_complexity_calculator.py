# -*- coding: utf-8 -*-

"""A simple script for computing child model parameters and controller search space
complexities
ZZJ, 11.19.2019
"""

import numpy as np


def child_model_params(num_features, num_layers, max_units):
    c = num_features * num_layers * max_units + (max_units * num_layers) ** 2 / 2
    return c


def controller_search_space(input_blocks, output_blocks, num_layers, num_choices_per_layer):
    s = np.log10(num_choices_per_layer) * num_layers
    s += np.log10(2) * (num_layers-1)*num_layers/2
    s += np.log10(input_blocks) * num_layers + np.log10(output_blocks) * num_layers
    return s
