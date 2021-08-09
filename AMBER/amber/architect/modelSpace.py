# -*- coding: UTF-8 -*-

"""
Model space to perform architecture search
"""

# Author       : ZZJ
# Initial Date : Nov. 17, 2018
# Last Update  : Aug. 16, 2020

from __future__ import print_function

from collections import defaultdict

import numpy as np


def get_layer_shortname(layer):
    """Get the short name for a computational operation of a layer, useful in converting a Layer object to a string as
    ID or when plotting

    Parameters
    ----------
    layer : amber.architect.Operation
        The ``Operation`` object for any layer.

    Returns
    -------
    sn : str
        The unique short name for this operation

    TODO
    -----
    Consider refactoring ``layer`` to ``operation``
    """
    if layer.Layer_type == 'conv1d':
        sn = "conv_f%s_k%s_%s" % (layer.Layer_attributes['filters'], layer.Layer_attributes['kernel_size'],
                                  layer.Layer_attributes['activation'])
        if 'dilation' in layer.Layer_attributes:
            sn += '_d%i' % layer.Layer_attributes['dilation']
    elif layer.Layer_type == 'denovo':
        sn = "%s_f%s_k%s" % ('regconv2d', layer.Layer_attributes['filters'], layer.Layer_attributes['kernel_size'])
        # sn = "%s_f%s_k%s"%('denovo', layer.Layer_attributes['filters'], layer.Layer_attributes['kernel_size'])

    elif layer.Layer_type == 'dense':
        sn = "%s_u%s_%s" % (layer.Layer_type, layer.Layer_attributes['units'], layer.Layer_attributes['activation'])

    elif layer.Layer_type == 'maxpool1d' or layer.Layer_type == 'avgpool1d':
        sn = layer.Layer_type

    elif layer.Layer_type == 'flatten' or layer.Layer_type == 'identity' or layer.Layer_type == 'globalmaxpool1d' or layer.Layer_type == 'globalavgpool1d':
        sn = layer.Layer_type

    elif layer.Layer_type == 'sfc':
        sn = layer.Layer_type

    else:
        sn = str(layer)
    return sn


class State(object):
    """The Amber internal holder for a computational operation at any layer

    Parameters
    ----------
    Layer_type : str
        The string for the operation type; supports most commonly used ``tf.keras.layers`` types

    kwargs :
        Operation/layer specifications are parsed through keyword arguments

    Attributes
    ----------
    Layer_type : str
        The string for the operation type.

    Layer_attributes : dict
        The dictionary that holds key-value pairs for all specification for this layer.

    Notes
    ------
    Any attributes that are not specified in ``Layer_attributes`` will use the default value as defined in
    ``tf.keras.layers``. For example, if you do not specify ``activation`` in ``Layer_attributes``, it will use ``linear``.

    Examples
    --------
    For example, to create a 1D-convolutional operation with ReLU activation, kernel size=8, number of kernels=32::

        >>> from amber.architect import State
        >>> op = State("conv1d", filters=32, kernel_size=8, activation='relu')

    """

    def __init__(self, Layer_type, **kwargs):
        Layer_type = Layer_type.lower()
        # assert Layer_type in [
        #    'conv1d', 'maxpool1d', 'avgpool1d',
        #    'conv2d', 'maxpool2d', 'avgpool2d',
        #    'lstm',
        #    'dense', 'input', 'identity',
        #    'dropout', 'sparsek_vec', 'batchnorm',
        #    'flatten', 'globalavgpool1d', 'globalavgpool2d', 'globalmaxpool1d', 'globalmaxpool1d',
        #    'data', 'denovo', 'sfc',
        #    'concatenate'
        #]

        self.Layer_type = Layer_type
        self.Layer_attributes = kwargs

    def __str__(self):
        return "{}:{}".format(self.Layer_type, self.Layer_attributes)

    def __eq__(self, other):
        return self.Layer_type == other.Layer_type and self.Layer_attributes == other.Layer_attributes

    def __hash__(self):
        unroll_attr = ((x, self.Layer_attributes[x]) for x in self.Layer_attributes)
        return hash((self.Layer_type, unroll_attr))


class ModelSpace:
    """Model Space constructor

    Provides utility functions for holding "states" / "operations" that the controller must use to train and predict.
    Also provides a more convenient way to define the model search space

    There are several ways to construct a model space. For example, one way is to initialize an empty ``ModelSpace`` then
    iteratively add layers to it, where each layer has a number of candidate operations::

        >>> def get_model_space(out_filters=64, num_layers=9):
        >>>    model_space = ModelSpace()
        >>>    num_pool = 4
        >>>    expand_layers = [num_layers//num_pool*i-1 for i in range(1, num_pool)]
        >>>    for i in range(num_layers):
        >>>        model_space.add_layer(i, [
        >>>            Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu'),
        >>>            Operation('conv1d', filters=out_filters, kernel_size=4, activation='relu'),
        >>>            Operation('maxpool1d', filters=out_filters, pool_size=4, strides=1),
        >>>            Operation('avgpool1d', filters=out_filters, pool_size=4, strides=1),
        >>>            Operation('identity', filters=out_filters),
        >>>      ])
        >>>        if i in expand_layers:
        >>>            out_filters *= 2
        >>>    return model_space

    Alternatively, ModelSpace can also be constructed from a dictionary.

    """

    def __init__(self, **kwargs):
        self.state_space = defaultdict(list)

    def __str__(self):
        return "StateSpace with {} layers and {} total combinations".format(len(self.state_space),
                                                                            self.get_space_size())

    def __len__(self):
        return len(self.state_space)

    def __getitem__(self, layer_id):
        if layer_id < 0:
            layer_id = len(self.state_space) + layer_id
        if layer_id not in self.state_space:
            raise IndexError('layer_id out of range')
        return self.state_space[layer_id]

    def __setitem__(self, layer_id, layer_states):
        self.add_layer(layer_id, layer_states)

    def get_space_size(self):
        """Get the total model space size by the product of all candidate operations across all layers. No residual
        connections are considered.

        Returns
        -------
        size : int
            The total number of possible combinations of operations.
        """
        size_ = 1
        for i in self.state_space:
            size_ *= len(self.state_space[i])
        return size_

    def add_state(self, layer_id, state):
        """Append a new state/operation to a layer

        Parameters
        ----------
        layer_id : int
            Which layer to append a new operation.

        state : amber.architect.State
            The new operation object to be appended.

        Returns
        -------

        """
        self.state_space[layer_id].append(state)

    def delete_state(self, layer_id, state_id):
        """Delete an operation from layer

        Parameters
        ----------
        layer_id : int
            Which layer to delete an operation

        state_id : int
            Which operation index to be deleted

        Returns
        -------

        """
        del self.state_space[layer_id][state_id]

    def add_layer(self, layer_id, layer_states=None):
        """Add a new layer to model space

        Parameters
        ----------
        layer_id : int
            The layer id of which layer to be added. Can be incontinuous to previous layers.

        layer_states : list of amber.architect.Operation
            A list of ``Operation`` object to be added.

        Returns
        -------
        bool
            Boolean value of Whether the model space is valid after inserting this layer
        """
        if layer_states is None:
            self.state_space[layer_id] = []
        else:
            self.state_space[layer_id] = layer_states
        return self._check_space_integrity()

    def delete_layer(self, layer_id):
        """Delete an entire layer and its associated values

        Parameters
        ----------
        layer_id : int
            which layer index to be deleted

        Returns
        -------
        bool
            Boolean value of Whether the model space is valid after inserting this layer
        """
        del self.state_space[layer_id]
        return self._check_space_integrity()

    def _check_space_integrity(self):
        return len(self.state_space) - 1 == max(self.state_space.keys())

    def print_state_space(self):
        """
        print out the model space in a nice layout (not so nice yet)
        """
        for i in range(len(self.state_space)):
            print("Layer {}".format(i))
            print("\n".join(["  " + str(x) for x in self.state_space[i]]))
            print('-' * 10)
        return

    def get_random_model_states(self):
        """Get a random combination of model operations throughout each layer

        Returns
        -------
        model_states : list
            A list of randomly sampled model operations
        """
        model_states = []
        for i in range(len(self.state_space)):
            model_states.append(np.random.choice(self.state_space[i]))
        return model_states

    @staticmethod
    def from_dict(d):
        """Static method for creating a ModelSpace from a Dictionary or List

        Parameters
        ----------
        d : dict or list
            A dictionary or list specifying candidate operations for each layer

        Returns
        -------
        amber.architect.ModelSpace
            The constructed model space from the given dict/list

        """
        import ast
        assert type(d) in (dict, list)
        num_layers = len(d)
        ms = ModelSpace()
        for i in range(num_layers):
            for j in range(len(d[i])):
                if 'shape' in d[i][j] and type(d[i][j]['shape']) is str:
                    d[i][j] = ast.literal_eval(d[i][j]['shape'])

            ms.add_layer(layer_id=i, layer_states=[State(**d[i][j]) for j in range(len(d[i]))])
        return ms


class BranchedModelSpace(ModelSpace):
    """
    Parameters
    ----------
    subspaces : list
        A list of `ModelSpace`. First element is a list of input branches. Second element is a stem model space
    concat_op : str
        string identifier for how to concatenate different input branches

    """
    def __init__(self, subspaces, concat_op='concatenate', **kwargs):
        super().__init__(**kwargs)
        self.subspaces = subspaces
        self.concat_op = concat_op
        # layer id to branch; expects a tuple of two elements
        # first element is type index, 0=input branch, 1=stem
        # second element is branch index, int=index of list, None=only one space present
        self._layer_to_branch = {}
        self._branch_to_layer = {}
        # delineate subspaces
        layer_id = 0
        for i, model_space in enumerate(self.subspaces[0]):
            for _layer in range(len(model_space)):
                self.state_space[layer_id] = model_space[_layer]
                self._layer_to_branch[layer_id] = (0, i)
                layer_id += 1
        for _layer in range(len(self.subspaces[1])):
            self.state_space[layer_id] = self.subspaces[1][_layer]
            self._layer_to_branch[layer_id] = (1, None)
            layer_id += 1
        for k, v in self._layer_to_branch.items():
            if v in self._branch_to_layer:
                self._branch_to_layer[v].append(k)
            else:
                self._branch_to_layer[v] = [k]

    @property
    def layer_to_branch(self):
        return self._layer_to_branch

    @property
    def branch_to_layer(self):
        return self._branch_to_layer


# alias
Operation = State
