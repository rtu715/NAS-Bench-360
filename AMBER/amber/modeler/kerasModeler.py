from abc import ABC

import tensorflow.keras as keras
from ..architect import Operation
from .dag import get_layer
import numpy as np
from .enasModeler import ModelBuilder
import tensorflow as tf
from .architectureDecoder import MultiIOArchitecture, ResConvNetArchitecture
from tensorflow.keras.layers import Concatenate, Add, Dense, Conv1D, MaxPooling1D, AveragePooling1D, \
    GlobalAveragePooling1D, Flatten, BatchNormalization, LeakyReLU, Dropout, Activation, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras.models import Model
import copy
from ..architect.modelSpace import BranchedModelSpace


class KerasModelBuilder(ModelBuilder):
    def __init__(self, inputs_op, output_op, model_compile_dict, model_space=None, gpus=None, **kwargs):
        self.model_compile_dict = model_compile_dict
        self.input_node = inputs_op
        self.output_node = output_op
        self.model_space = model_space
        self.gpus = gpus

    def __call__(self, model_states):
        if self.gpus is None or self.gpus == 1:
            model = build_sequential_model(
                        model_states=model_states,
                        input_state=self.input_node,
                        output_state=self.output_node,
                        model_compile_dict=self.model_compile_dict,
                        model_space=self.model_space
                        )
        elif type(self.gpus) is int:
            model = build_multi_gpu_sequential_model(
                        model_states=model_states,
                        input_state=self.input_node,
                        output_state=self.output_node,
                        model_compile_dict=self.model_compile_dict,
                        model_space=self.model_space,
                        gpus=self.gpus
                        )
        elif type(self.gpus) is list:
            mirrored_strategy = tf.distribute.MirroredStrategy(devices=self.gpus)
            with mirrored_strategy.scope():
                model = build_sequential_model(
                            model_states=model_states,
                            input_state=self.input_node,
                            output_state=self.output_node,
                            model_compile_dict=self.model_compile_dict,
                            model_space=self.model_space
                            )
        return model


class KerasBranchModelBuilder(ModelBuilder):
    def __init__(self, inputs_op, output_op, model_compile_dict, model_space=None, with_bn=False, **kwargs):
        assert type(model_space) is BranchedModelSpace
        assert len(inputs_op) == len(model_space.subspaces[0])
        self.inputs_op = inputs_op
        self.output_op = output_op
        self.model_space = model_space
        self.model_compile_dict = model_compile_dict
        self.with_bn = with_bn
        self._branch_to_layer = self.model_space.branch_to_layer

    def _build_branch(self, input_op, model_states, model_space):
        if issubclass(type(input_op), Operation):
            inp = get_layer(None, input_op)
        else:
            inp = input_op
        x = inp
        assert len(model_states) > 0
        for i, state in enumerate(model_states):
            if issubclass(type(state), Operation):
                x = get_layer(x, state)
            elif issubclass(type(state), int) or np.issubclass_(type(state), np.integer):
                assert model_space is not None, "if provided integer model_arc, must provide model_space in kwargs"
                x = get_layer(x, model_space[i][state], with_bn=self.with_bn)
            else:
                raise Exception("cannot understand %s of type %s" % (state, type(state)))
        return inp, x

    def __call__(self, model_states, **kwargs):
        inps = []
        branches = []
        # build branch sequentially
        for i in range(len(self.inputs_op)):
            inp, out = self._build_branch(
                input_op=self.inputs_op[i],
                model_states=[model_states[j] for j in self._branch_to_layer[(0, i)]],
                model_space=self.model_space.subspaces[0][i]
            )
            inps.append(inp)
            branches.append(out)
        # merge branches
        if self.model_space.concat_op == 'concatenate':
            branch_merge = get_layer(x=branches, state=Operation('concatenate'))
        else:
            raise ValueError('Model builder cannot understand model space concat op: %s' % self.model_space.conat_op)
        # build stem
        _, h = self._build_branch(
            input_op=branch_merge,
            model_states=[model_states[j] for j in self._branch_to_layer[(1, None)]],
            model_space=self.model_space.subspaces[1]
        )
        out = get_layer(x=h, state=self.output_op)
        model = Model(inputs=inps, outputs=out)
        model.compile(**self.model_compile_dict)
        return model


class KerasResidualCnnBuilder(ModelBuilder):
    """Function class for converting an architecture sequence tokens to a Keras model

    Parameters
    ----------
    inputs_op : amber.architect.modelSpace.Operation
    output_op : amber.architect.modelSpace.Operation
    fc_units : int
        number of units in the fully-connected layer
    flatten_mode : {'GAP', 'Flatten'}
        the flatten mode to convert conv layers to fully-connected layers.
    model_compile_dict : dict
    model_space : amber.architect.modelSpace.ModelSpace
    dropout_rate : float
        dropout rate, must be 0<dropout_rate<1
    wsf : int
        width scale factor
    """
    def __init__(self, inputs_op, output_op, fc_units, flatten_mode, model_compile_dict, model_space,
                 dropout_rate=0.2, wsf=1, add_conv1_under_pool=True, verbose=1, **kwargs):
        self.model_compile_dict = model_compile_dict
        self.inputs = inputs_op
        self.outputs = output_op
        self.fc_units = fc_units
        self.verbose = verbose
        assert flatten_mode.lower() in {'gap', 'flatten'}, "Unknown flatten mode: %s" % flatten_mode
        self.flatten_mode = flatten_mode.lower()
        self.model_space = model_space
        self.dropout_rate = dropout_rate
        self.wsf = wsf
        self.add_conv1_under_pool = add_conv1_under_pool
        self.decoder = ResConvNetArchitecture(model_space=model_space)

    def __call__(self, model_states):
        model = self._convert(model_states, verbose=self.verbose)
        if model is not None:
            model.compile(**self.model_compile_dict)
        return model

    def _convert(self, arc_seq, verbose=True):
        out_filters, pool_layers = self.get_out_filters(self.model_space)

        inp = get_layer(x=None, state=self.inputs)
        # this is assuming all choices have the same out_filters
        stem_conv = Operation('conv1d', kernel_size=8, filters=out_filters[0], activation="linear")
        x = self.res_layer(stem_conv, self.wsf, inp, name="stem_conv",
                           add_conv1_under_pool=self.add_conv1_under_pool)

        start_idx = 0
        layers = []
        for layer_id in range(len(self.model_space)):
            if verbose:
                print("start_idx=%i, layer id=%i, out_filters=%i x %i" % (
                    start_idx, layer_id, out_filters[layer_id], self.wsf))
            count = arc_seq[start_idx]
            this_layer = self.model_space[layer_id][count]
            if verbose: print(this_layer)
            if layer_id == 0:
                x = self.res_layer(this_layer, self.wsf, x, name="L%i" % layer_id,
                                   add_conv1_under_pool=self.add_conv1_under_pool)
            else:
                x = self.res_layer(this_layer, self.wsf, layers[-1], name="L%i" % layer_id,
                                   add_conv1_under_pool=self.add_conv1_under_pool)

            if layer_id > 0:
                skip = arc_seq[start_idx + 1: start_idx + layer_id + 1]
                skip_layers = [layers[i] for i in range(len(layers)) if skip[i] == 1]
                if verbose: print("skip=%s" % skip)
                if len(skip_layers):
                    skip_layers.append(x)
                    x = Add(name="L%i_resAdd" % layer_id)(skip_layers)
                x = BatchNormalization(name="L%i_resBn" % layer_id)(x)

            if self.dropout_rate != 0:
                x = Dropout(self.dropout_rate, name="L%i_dropout" % layer_id)(x)

            layers.append(x)
            if layer_id in pool_layers:
                pooled_layers = []
                for i, layer in enumerate(layers):
                    pooled_layers.append(
                        self.factorized_reduction_layer(
                            layer,
                            out_filters[layer_id + 1] * self.wsf,
                            name="pool_at_%i_from_%i" % (layer_id, i))
                    )
                if verbose: print("pooled@%i, %s" % (layer_id, pooled_layers))
                layers = pooled_layers

            start_idx += 1 + layer_id
            if verbose: print('-' * 80)

        # fully-connected layer
        if self.flatten_mode == 'gap':
            x = GlobalAveragePooling1D()(x)
        elif self.flatten_mode == 'flatten':
            x = Flatten()(x)
        else:
            raise Exception("Unknown flatten mode: %s" % self.flatten_mode)
        if self.dropout_rate != 0:
            x = Dropout(self.dropout_rate)(x)
        x = Dense(units=self.fc_units, activation="relu")(x)

        out = get_layer(x=x, state=self.outputs)

        model = Model(inputs=inp, outputs=out)
        return model

    @staticmethod
    def factorized_reduction_layer(inp, out_filter, name, reduction_factor=4):
        x = Conv1D(out_filter,
                   kernel_size=1,
                   strides=1,
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding="same",
                   name=name
                   )(inp)
        x = MaxPooling1D(pool_size=reduction_factor, strides=reduction_factor, padding="same")(x)
        return x

    @staticmethod
    def res_layer(layer, width_scale_factor, inputs, l2_reg=5e-7, name="layer", add_conv1_under_pool=True):
        if layer.Layer_type == 'conv1d':
            activation = layer.Layer_attributes['activation']
            num_filters = width_scale_factor * layer.Layer_attributes['filters']
            kernel_size = layer.Layer_attributes['kernel_size']
            if 'dilation' in layer.Layer_attributes:
                dilation = layer.Layer_attributes['dilation']
            else:
                dilation = 1
            x = Conv1D(num_filters,
                       kernel_size=kernel_size,
                       strides=1,
                       padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(l2_reg),
                       kernel_constraint=constraints.max_norm(0.9),
                       use_bias=False,
                       name="%s_conv" % name if dilation == 1 else "%s_conv_d%i" % (name, dilation),
                       dilation_rate=dilation
                       )(inputs)
            x = BatchNormalization(name="%s_bn" % name)(x)
            if activation in ("None", "linear"):
                pass
            elif activation in ("relu", "sigmoid", "tanh", "softmax", "elu"):
                x = Activation(activation, name="%s_%s" % (name, activation))(x)
            elif activation == "leaky_relu":
                x = LeakyReLU(alpha=0.2, name="%s_%s" % (name, activation))(x)
            else:
                raise Exception("Unknown activation: %s" % activation)
        elif layer.Layer_type == 'maxpool1d' or layer.Layer_type == 'avgpool1d':
            num_filters = width_scale_factor * layer.Layer_attributes['filters']
            pool_size = layer.Layer_attributes['pool_size']
            if add_conv1_under_pool:
                x = Conv1D(num_filters,
                           kernel_size=1,
                           strides=1,
                           padding='same',
                           kernel_initializer='he_normal',
                           use_bias=False,
                           name="%s_maxpool_conv" % name
                           )(inputs)
                x = BatchNormalization(name="%s_bn" % name)(x)
                x = Activation("relu", name="%s_relu" % name)(x)
            else:
                x = inputs

            if layer.Layer_type == 'maxpool1d':
                x = MaxPooling1D(pool_size=pool_size, strides=1, padding='same', name="%s_maxpool" % name)(x)
            elif layer.Layer_type == 'avgpool1d':
                x = AveragePooling1D(pool_size=pool_size, strides=1, padding='same', name="%s_avgpool" % name)(x)
            else:
                raise Exception("Unknown pool: %s" % layer.Layer_type)

        elif layer.Layer_type == 'identity':
            x = Lambda(lambda t: t, name="%s_id" % name)(inputs)
        else:
            raise Exception("Unknown type: %s" % layer.Layer_type)
        return x

    @staticmethod
    def get_out_filters(model_space):
        out_filters = []
        pool_layers = []
        for layer_id in range(len(model_space)):
            layer = model_space[layer_id]
            this_out_filters = [l.Layer_attributes['filters'] for l in layer]
            assert len(
                set(this_out_filters)) == 1, "EnasConv1dDAG only supports one identical number of filters per layer," \
                                             "but found %i in layer %s" % (len(set(this_out_filters)), layer)
            if len(out_filters) and this_out_filters[0] != out_filters[-1]:
                pool_layers.append(layer_id - 1)

            out_filters.append(this_out_filters[0])
        # print(out_filters)
        # print(pool_layers)
        return out_filters, pool_layers


class KerasMultiIOModelBuilder(ModelBuilder):
    """
    Note:
        Still not working if num_outputs=0
    """
    def __init__(self, inputs_op, output_op, model_compile_dict, model_space, with_input_blocks, with_output_blocks, dropout_rate=0.2, wsf=1, **kwargs):
        self.model_compile_dict = model_compile_dict
        self.inputs = inputs_op
        self.outputs = output_op
        self.model_space = model_space
        self.num_inputs = len(inputs_op) if type(inputs_op) in (list, tuple) else 0
        self.num_outputs = len(output_op) if type(output_op) in (list, tuple) else 0
        assert not (self.num_inputs==0 & self.num_outputs==0), "MultiIO cannot have single input and single output at " \
                                                               "the same time "
        self.with_input_blocks = with_input_blocks
        self.with_output_blocks = with_output_blocks
        if self.with_input_blocks: assert self.num_inputs > 0, "you specified with_input_blocks=True for " \
                                                               "KerasMultiIOModelBuilder, but only provided 1 " \
                                                               "num_inputs "
        self.decoder = MultiIOArchitecture(num_layers=len(self.model_space), num_inputs=self.num_inputs*self.with_input_blocks, num_outputs=self.num_outputs*self.with_output_blocks)

    def __call__(self, model_states):
        model = self._convert(model_states)
        if model is not None:
            model.compile(**self.model_compile_dict)
        return model

    def _convert(self, arc_seq, with_bn=True, wsf=1):
        inputs = [get_layer(x=None, state=x) for x in self.inputs] if self.num_inputs>0 else [get_layer(x=None, state=self.inputs)]
        op, inp, skp, out = self.decoder.decode(arc_seq)
        out_rowsum = np.apply_along_axis(np.sum, 1, out)
        out_colsum = np.apply_along_axis(np.sum, 0, out)
        skp_rowsum = np.array([1] + [sum(x) for x in skp])
        with_input_blocks = self.with_input_blocks
        # missing output connection
        if any(out_rowsum==0):
            print("invalid model: unconnected output")
            return None
        # missing output with skip connection
        if self.with_input_blocks is False and any( (skp_rowsum==0)&(out_colsum!=0) ):
            print("invalid model: output connected to layer with no input")
            return None

        # Build the model until outputs
        prev_layers = []
        for layer_id in range(len(self.model_space)):
            this_op = op[layer_id]
            # Prepare the inputs
            if with_input_blocks:
                this_inputs = [inputs[i] for i in np.where(inp[layer_id])[0]]
            else:
                this_inputs = inputs if layer_id == 0 else []
            if layer_id > 0:
                this_inputs += [ prev_layers[i] for i in np.where(skp[layer_id-1])[0] if prev_layers[i] is not None ]

            # Connect tensors
            model_op = copy.deepcopy(self.model_space[layer_id][this_op])
            if 'units' in model_op.Layer_attributes:
                model_op.Layer_attributes['units'] *= wsf
            elif 'filters' in model_op.Layer_attributes:
                model_op.Layer_attributes['filters'] *= wsf
            else:
                raise Exception("Cannot use wsf")
 
            if len(this_inputs) > 1:
                input_tensor = Concatenate()(this_inputs)
                layer = get_layer(x=input_tensor, state=model_op, with_bn=with_bn)
                prev_layers.append(layer)
            elif len(this_inputs) == 1:
                input_tensor = this_inputs[0]
                layer = get_layer(x=input_tensor, state=model_op, with_bn=with_bn)
                prev_layers.append(layer)
            else:
                prev_layers.append(None)  # skipped a layer

        # Build the outputs
        outputs_inputs = []
        for m, o in enumerate(out):
            idx = [i for i in np.where(o)[0] if prev_layers[i] is not None]
            if len(idx) > 1:
                outputs_inputs.append( Concatenate()([prev_layers[i] for i in idx])  )
            elif len(idx) == 1:
                outputs_inputs.append(prev_layers[idx[0]] )
            else:
                #raise Exception("Unconnected output %i"%m)
                print("Secondary unconnected output %i"%m)
                return None
        outputs = [get_layer(x=outputs_inputs[i], state=self.outputs[i]) for i in range(self.num_outputs)  ]
        model = Model(inputs=inputs, outputs=outputs)
        return model
     

def build_sequential_model(model_states, input_state, output_state, model_compile_dict, **kwargs):
    """
    Parameters
    ----------
    model_states: a list of _operators sampled from operator space
    input_state:
    output_state: specifies the output tensor, e.g. Dense(1, activation='sigmoid')
    model_compile_dict: a dict of `loss`, `optimizer` and `metrics`

    Returns
    ---------
    Keras.Model
    """
    inp = get_layer(None, input_state)
    x = inp
    model_space = kwargs.pop("model_space", None)
    for i, state in enumerate(model_states):
        if issubclass(type(state), Operation):
            x = get_layer(x, state)
        elif issubclass(type(state), int) or np.issubclass_(type(state), np.integer):
            assert model_space is not None, "if provided integer model_arc, must provide model_space in kwargs"
            x = get_layer(x, model_space[i][state])
        else:
            raise Exception("cannot understand %s of type %s" % (state, type(state)))
    out = get_layer(x, output_state)
    model = Model(inputs=inp, outputs=out)
    if not kwargs.pop('stop_compile', False):
        model.compile(**model_compile_dict)
    return model


def build_multi_gpu_sequential_model(model_states, input_state, output_state, model_compile_dict, gpus=4, **kwargs):
    try:
        from tensorflow.keras.utils import multi_gpu_model
    except Exception as e:
        raise Exception("multi gpu not supported in keras. check your version. Error: %s" % e)
    with tf.device('/cpu:0'):
        vanilla_model = build_sequential_model(model_states, input_state, output_state, model_compile_dict, stop_compile=True, **kwargs)
    model = multi_gpu_model(vanilla_model, gpus=gpus)
    model.compile(**model_compile_dict)
    return model


def build_sequential_model_from_string(model_states_str, input_state, output_state, state_space, model_compile_dict):
    """build a sequential model from a string of states
    """
    assert len(model_states_str) == len(state_space)
    str_to_state = [[str(state) for state in state_space[i]] for i in range(len(state_space))]
    try:
        model_states = [state_space[i][str_to_state[i].index(model_states_str[i])] for i in range(len(state_space))]
    except ValueError:
        raise Exception("model_states_str not found in state-space")
    return build_sequential_model(model_states, input_state, output_state, model_compile_dict)


def build_multi_gpu_sequential_model_from_string(model_states_str, input_state, output_state, state_space,
                                                 model_compile_dict):
    """build a sequential model from a string of states
    """
    assert len(model_states_str) == len(state_space)
    str_to_state = [[str(state) for state in state_space[i]] for i in range(len(state_space))]
    try:
        model_states = [state_space[i][str_to_state[i].index(model_states_str[i])] for i in range(len(state_space))]
    except ValueError:
        raise Exception("model_states_str not found in state-space")
    return build_multi_gpu_sequential_model(model_states, input_state, output_state, model_compile_dict)
