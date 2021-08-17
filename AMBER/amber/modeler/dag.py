"""represent neural network computation graph
as a directed-acyclic graph from a list of 
architecture selections

Notes
-----
this is an upgrade of the `NetworkManager` class
"""

# Author: ZZJ
# Initial Date: June 12, 2019
# Last update:  Aug. 18, 2020

import numpy as np
import warnings
from ..utils import corrected_tf as tf
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
# TODO: need to clean up `State` as `Operation`
from ..architect.modelSpace import State

# for general child
from .child import DenseAddOutputChild, EnasAnnModel, EnasCnnModel
from ..architect.commonOps import get_tf_metrics, get_keras_train_ops, get_tf_layer, get_tf_loss, create_weight, \
    create_bias, batch_norm1d
# for get layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, GaussianNoise
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Input, Lambda, Permute, BatchNormalization, Activation
from tensorflow.keras.layers import LSTM
from ._operators import Layer_deNovo, SeparableFC, sparsek_vec
from ..architect.modelSpace import get_layer_shortname


def get_dag(arg):
    """Getter method for getting a DAG class from a string

    DAG refers to the underlying tensor computation graphs for child models. Whenever possible, we prefer to use Keras
    Model API to get the job done. For ENAS, the parameter-sharing scheme is implemented by tensorflow.

    Parameters
    ----------
    arg : str or callable
        return the DAG constructor corresponding to that identifier; if is callable, assume it's a DAG constructor
        already, do nothing and return it

    Returns
    -------
    callable
        A DAG constructor
    """
    if arg is None:
        return None
    elif type(arg) is str:
        if arg.lower() == 'dag':
            return DAG
        elif arg.lower() == 'inputblockdag':
            return InputBlockDAG
        elif arg.lower() == 'inputblockauxlossdag':
            return InputBlockAuxLossDAG
        elif arg.lower() == 'enasanndag':
            return EnasAnnDAG
        elif arg.lower() == 'enasconv1ddag':
            return EnasConv1dDAG
        elif arg == 'EnasConv1DwDataDescrption':
            return EnasConv1DwDataDescrption
    elif callable(arg):
        return arg
    else:
        raise ValueError("Could not understand the DAG func:", arg)


def get_layer(x, state, with_bn=False):
    """Getter method for a Keras layer, including native Keras implementation and custom layers that are not included in
    Keras.

    Parameters
    ----------
    x : tf.keras.layers or None
        The input Keras layer
    state : amber.architect.Operation
        The target layer to be built
    with_bn : bool, optional
        If true, add batch normalization layers before activation

    Returns
    -------
    x : tf.keras.layers
        The built target layer connected to input x
    """
    if state.Layer_type == 'dense':
        if with_bn is True:
            actv_fn = state.Layer_attributes.pop('activation', 'linear')
            x = Dense(**state.Layer_attributes)(x)
            x = BatchNormalization()(x)
            x = Activation(actv_fn)(x)
            return x
        else:
            return Dense(**state.Layer_attributes)(x)

    elif state.Layer_type == 'sfc':
        return SeparableFC(**state.Layer_attributes)(x)

    elif state.Layer_type == 'input':
        return Input(**state.Layer_attributes)

    elif state.Layer_type == 'conv1d':
        if with_bn is True:
            actv_fn = state.Layer_attributes.pop('activation', 'linear')
            x = Conv1D(**state.Layer_attributes)(x)
            x = BatchNormalization()(x)
            x = Activation(actv_fn)(x)
            return x
        else:
            return Conv1D(**state.Layer_attributes)(x)

    elif state.Layer_type == 'denovo':
        x = Lambda(lambda t: K.expand_dims(t))(x)
        x = Permute(dims=(2, 1, 3))(x)
        x = Layer_deNovo(**state.Layer_attributes)(x)
        x = Lambda(lambda t: K.squeeze(t, axis=1))(x)
        return x

    elif state.Layer_type == 'sparsek_vec':
        x = Lambda(sparsek_vec, **state.Layer_attributes)(x)
        return x

    elif state.Layer_type == 'maxpool1d':
        return MaxPooling1D(**state.Layer_attributes)(x)

    elif state.Layer_type == 'avgpool1d':
        return AveragePooling1D(**state.Layer_attributes)(x)

    elif state.Layer_type == 'lstm':
        return LSTM(**state.Layer_attributes)(x)

    elif state.Layer_type == 'flatten':
        return Flatten()(x)

    elif state.Layer_type == 'globalavgpool1d':
        return GlobalAveragePooling1D()(x)

    elif state.Layer_type == 'globalmaxpool1d':
        return GlobalMaxPooling1D()(x)

    elif state.Layer_type == 'dropout':
        return Dropout(**state.Layer_attributes)(x)

    elif state.Layer_type == 'identity':
        return Lambda(lambda t: t, **state.Layer_attributes)(x)

    elif state.Layer_type == 'gaussian_noise':
        return GaussianNoise(**state.Layer_attributes)(x)

    elif state.Layer_type == 'concatenate':
        return Concatenate(**state.Layer_attributes)(x)

    else:
        raise Exception('Layer_type "%s" is not understood' % state.Layer_type)


class ComputationNode:
    def __init__(self, operation, node_name, merge_op=Concatenate):
        assert type(operation) is State, "Expect operation is of type amber.architect.State, got %s" % type(
            operation)
        self.operation = operation
        self.node_name = node_name
        self.merge_op = merge_op
        self.parent = []
        self.child = []
        self.operation_layer = None
        self.merge_layer = None
        self.is_built = False

    def build(self):
        """Build the keras layer with merge operations if applicable

        Notes
        -----
        when building a node, its parents must all be built already
        """
        if self.parent:
            if len(self.parent) > 1:
                self.merge_layer = self.merge_op()([p.operation_layer for p in self.parent])
            else:
                self.merge_layer = self.parent[0].operation_layer
        self.operation.Layer_attributes['name'] = self.node_name
        self.operation_layer = get_layer(self.merge_layer, self.operation)
        self.is_built = True


class DAG:
    def __init__(self, arc_seq, num_layers, model_space, input_node, output_node,
                 with_skip_connection=True,
                 with_input_blocks=True):
        assert all([not x.is_built for x in input_node]), "input_node must not have been built"
        if type(output_node) is list:
            assert all([not x.is_built for x in output_node]), "output_node must not have been built"

            # seems this Base Class only handles a single output node.. so need to update here
            # TODO: for muliple input/output nodes, implement a new Class for graphing
            # ZZ, 2020.3.2
            assert len(output_node) == 1
            output_node = output_node[0]

        else:
            assert not output_node.is_built, "output_node must not have been built"
        self.arc_seq = np.array(arc_seq)
        self.num_layers = num_layers
        self.model_space = model_space
        self.input_node = input_node
        self.output_node = output_node
        self.with_skip_connection = with_skip_connection
        self.with_input_blocks = with_input_blocks
        self.model = None
        self.nodes = []

    def _build_dag(self):
        if self.with_input_blocks:
            assert type(self.input_node) in (list, tuple), "If ``with_input_blocks=True" \
                                                           "``, ``input_node`` must be " \
                                                           "array-like. " \
                                                           "Current type of input_node is %s and" \
                                                           " with_input_blocks=%s" % (
                                                               type(self.input_node), self.with_input_blocks)
        assert type(self.output_node) is ComputationNode

        nodes = self._init_nodes()
        nodes = self._prune_nodes(nodes)

        # build order is essential here
        node_list = self.input_node + nodes + [self.output_node]
        for node in node_list:
            try:
                node.build()
            except Exception as e:
                print(node.node_name)
                print([x.node_name for x in node.parent])
                print([x.node_name for x in node.child])
                raise e
        self.model = Model(inputs=[x.operation_layer for x in self.input_node],
                           outputs=[self.output_node.operation_layer])
        self.nodes = nodes
        return self.model

    def _init_nodes(self):
        """first read through the architecture sequence to initialize the nodes"""
        arc_pointer = 0
        nodes = []
        for layer_id in range(self.num_layers):
            arc_id = self.arc_seq[arc_pointer]
            op = self.model_space[layer_id][arc_id]
            parent = []
            node_ = ComputationNode(op, node_name="L%i_%s" % (layer_id, get_layer_shortname(op)))
            if self.with_input_blocks:
                inp_bl = np.where(self.arc_seq[arc_pointer + 1: arc_pointer + 1 + len(self.input_node)] == 1)[0]
                for i in inp_bl:
                    parent.append(self.input_node[i])
                    self.input_node[i].child.append(node_)
            else:
                if layer_id == 0:
                    for n in self.input_node:
                        n.child.append(node_)
                        parent.append(n)
            if self.with_skip_connection and layer_id > 0:
                skip_con = np.where(
                    self.arc_seq[arc_pointer + 1 + len(self.input_node) * self.with_input_blocks: arc_pointer + 1 + len(
                        self.input_node) * self.with_input_blocks + layer_id] == 1)[0]
                # print(layer_id, skip_con)
                for i in skip_con:
                    parent.append(nodes[i])
                    nodes[i].child.append(node_)
            else:
                if layer_id > 0:
                    parent.append(nodes[-1])
                    nodes[-1].child.append(node_)
                # leave first layer without parent, so it
                # will be connected to default input node
            node_.parent = parent
            nodes.append(node_)
            arc_pointer += 1 + int(self.with_input_blocks) * len(self.input_node) + int(
                self.with_skip_connection) * layer_id
        # print('initial', nodes)
        return nodes

    def _prune_nodes(self, nodes):
        """now need to deal with loose-ends: node with no parent, or no child
        """
        # CHANGE: add regularization to default input
        # such that all things in default_input would be deemed
        # un-important ZZ 2019.09.14
        # default_input_node = ComputationNode(State('Identity', name='default_input'), node_name="default_input")
        # default_input_node = ComputationNode(State('dropout', rate=0.999,  name='default_input'), node_name="default_input")
        # create an information bottleneck
        default_input_node = ComputationNode(State('dense', units=1, activation='linear', name='default_input'),
                                             node_name="default_input")
        # CHANGE: default input node cannot include every input node
        # otherwise will overwhelm the architecture. ZZ 2019.09.13
        # default_input_node.parent = self.input_node
        default_input_node.parent = [x for x in self.input_node if len(x.child) == 0]
        if default_input_node.parent:
            for x in self.input_node:
                if len(x.child) == 0:
                    x.child.append(default_input_node)
            has_default = True
        else:
            has_default = False
        is_default_intermediate = False
        # tmp_nodes: a tmp queue of connected/must-have nodes
        tmp_nodes = []
        for node in nodes:
            # filter out invalid parent nodes
            node.parent = [x for x in node.parent if x in tmp_nodes or x in self.input_node]
            # if no parent left, try to connect to default_input
            # otherwise, throw away as invalid
            if not node.parent:
                if has_default:
                    node.parent.append(default_input_node)
                    default_input_node.child.append(node)
                    is_default_intermediate = True
                else:
                    continue
            # if no child, connect to output
            if not node.child:
                self.output_node.parent.append(node)
                node.child.append(self.output_node)
            tmp_nodes.append(node)
        nodes = tmp_nodes
        # print('after filter', nodes)

        if has_default and not is_default_intermediate:
            default_input_node.child.append(self.output_node)
            self.output_node.parent.append(default_input_node)
            # for node in self.input_node:
            #    if not node.child:
            #        node.child.append(self.output_node)
            #        self.output_node.parent.append(node)
        if has_default:
            nodes = [default_input_node] + nodes
        return nodes


class InputBlockDAG(DAG):
    def __init__(self, add_output=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.with_input_blocks, "`InputBlockDAG` class only handles `with_input_blocks=True`"
        self.added_output_nodes = []
        self.add_output = add_output

    def _build_dag(self):
        assert type(self.input_node) in (list, tuple), "If ``with_input_blocks=True" \
                                                       "``, ``input_node`` must be " \
                                                       "array-like. " \
                                                       "Current type of input_node is %s and " \
                                                       "with_input_blocks=%s" % (
                                                           type(self.input_node), self.with_input_blocks)
        assert type(self.output_node) is ComputationNode

        nodes = self._init_nodes()
        nodes = self._prune_nodes(nodes)

        # build order is essential here
        node_list = self.input_node + nodes + self.added_output_nodes + [self.output_node]
        for node in node_list:
            try:
                node.build()
            except:
                print(node.node_name)
                print([x.node_name for x in node.parent])
                print([x.node_name for x in node.child])
                raise Exception('above')
        self.nodes = nodes
        self.model = DenseAddOutputChild(
            inputs=[x.operation_layer for x in self.input_node],
            outputs=[self.output_node.operation_layer] + [n.operation_layer for n in self.added_output_nodes],
            nodes=self.nodes
        )
        return self.model

    def _init_nodes(self):
        """first read through the architecture sequence to initialize the nodes,
        whenever a input block is connected, add an output tensor afterwards
        """
        arc_pointer = 0
        nodes = []
        for layer_id in range(self.num_layers):
            arc_id = self.arc_seq[arc_pointer]
            op = self.model_space[layer_id][arc_id]
            parent = []
            node_ = ComputationNode(op, node_name="L%i_%s" % (layer_id, get_layer_shortname(op)))
            inp_bl = np.where(self.arc_seq[arc_pointer + 1: arc_pointer + 1 + len(self.input_node)] == 1)[0]
            if any(inp_bl):
                for i in inp_bl:
                    parent.append(self.input_node[i])
                    self.input_node[i].child.append(node_)
                # do NOT add any additional outputs if this is
                #  the last layer..
                if self.add_output and layer_id != self.num_layers - 1:
                    if type(self.output_node) is list:
                        assert len(self.output_node) == 1
                        self.output_node = self.output_node[0]
                    new_out = ComputationNode(
                        operation=self.output_node.operation,
                        node_name="added_out_%i" % (len(self.added_output_nodes) + 1)
                    )
                    new_out.parent.append(node_)
                    self.added_output_nodes.append(new_out)
            if self.with_skip_connection and layer_id > 0:
                skip_con = np.where(
                    self.arc_seq[arc_pointer + 1 + len(self.input_node) * self.with_input_blocks: arc_pointer + 1 + len(
                        self.input_node) * self.with_input_blocks + layer_id] == 1)[0]
                # print(layer_id, skip_con)
                for i in skip_con:
                    parent.append(nodes[i])
                    nodes[i].child.append(node_)
            else:
                if layer_id > 0:
                    parent.append(nodes[-1])
                    nodes[-1].child.append(node_)
                # leave first layer without parent, so it
                # will be connected to default input node
            node_.parent = parent
            nodes.append(node_)
            arc_pointer += 1 + int(self.with_input_blocks) * len(self.input_node) + int(
                self.with_skip_connection) * layer_id
        # print('initial', nodes)
        return nodes


class InputBlockAuxLossDAG(InputBlockDAG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.add_output, "InputBlockAuxLossDAG must have `add_output=True`"
        # index is the List index in the output of `model.evaluate` method
        self.input_block_loss_mapping = {}

    def _build_dag(self):
        assert type(self.input_node) in (list, tuple), "If ``with_input_blocks=True" \
                                                       "``, ``input_node`` must be " \
                                                       "array-like. " \
                                                       "Current type of input_node is %s and with_input_blocks=%s" % (
                                                           type(self.input_node), self.with_input_blocks)
        assert type(self.output_node) is ComputationNode

        nodes = self._init_nodes()
        nodes = self._prune_nodes(nodes)

        # build order is essential here
        node_list = self.input_node + nodes + self.added_output_nodes + [self.output_node]
        for node in node_list:
            try:
                node.build()
            except Exception as e:
                print(node.node_name)
                print([x.node_name for x in node.parent])
                print([x.node_name for x in node.child])
                raise e
        self.nodes = nodes
        self.model = DenseAddOutputChild(
            inputs=[x.operation_layer for x in self.input_node],
            outputs=[self.output_node.operation_layer] + [n.operation_layer for n in self.added_output_nodes],
            nodes=self.nodes,
            # lcr: lowest common root
            block_loss_mapping=self.input_block_loss_mapping
        )
        return self.model

    def _aux_loss(self, nodes):
        input_blocks = [x.node_name for x in self.input_node]
        # inp_head keeps a queue of all leading branches for each input block
        inp_heads = {inp: None for inp in input_blocks}
        # inp_pair_root keeps the tree subroot for each pair of input blocks
        inp_pair_roots = {(b1, b2): 'None' for b1 in input_blocks for b2 in input_blocks}
        # root_leaves keeps all leaves for all nodes
        leaves_cardinality = {n.node_name: set([]) for n in nodes}
        for n in nodes:
            # get inputs to this layer and update heads
            _inputs = [x.node_name for x in n.parent if x.operation.Layer_type == 'input']
            inp_heads.update({x: n.node_name for x in _inputs})

            # get the set of parents nodes that are not input_blocks
            _ops = set([x.node_name for x in n.parent if x.operation.Layer_type != 'input'])

            # update leave cardinality
            for leaf in _inputs + [l for x in _ops for l in leaves_cardinality[x]]:
                leaves_cardinality[n.node_name].add(leaf)

            # update heads if connected to this layer
            inp_heads.update({x: n.node_name for x in input_blocks if inp_heads[x] in _ops})

            # update inp_pair_roots if new inp_heads met each other
            for b1 in input_blocks:
                for b2 in input_blocks:
                    if inp_pair_roots[(b1, b2)] != 'None':
                        continue
                    head1 = inp_heads[b1]
                    head2 = inp_heads[b2]
                    if head1 == head2 == n.node_name:
                        inp_pair_roots[(b1, b2)] = n.node_name

        aux_loss_nodes = []
        layer2loss = {}
        node_index = {node.node_name: node for node in nodes}
        for t in sorted(set(inp_pair_roots.values())):
            if t == 'None' or node_index[t] == nodes[-1]:
                layer2loss[t] = None  # None means look at the final loss, no aux loss
                continue
            else:
                new_out = ComputationNode(operation=self.output_node.operation,
                                          node_name="add_out_%i" % (len(aux_loss_nodes) + 1))
                new_out.parent.append(node_index[t])
                aux_loss_nodes.append(new_out)
                layer2loss[t] = len(aux_loss_nodes) + 1
        self.added_output_nodes = aux_loss_nodes

        for b1, b2 in inp_pair_roots:
            self.input_block_loss_mapping[(b1, b2)] = layer2loss[inp_pair_roots[(b1, b2)]]
        return

    def _init_nodes(self):
        arc_pointer = 0
        nodes = []
        for layer_id in range(self.num_layers):
            arc_id = self.arc_seq[arc_pointer]
            op = self.model_space[layer_id][arc_id]
            parent = []
            node_ = ComputationNode(op, node_name="L%i_%s" % (layer_id, get_layer_shortname(op)))
            inp_bl = np.where(self.arc_seq[arc_pointer + 1: arc_pointer + 1 + len(self.input_node)] == 1)[0]
            if any(inp_bl):
                for i in inp_bl:
                    parent.append(self.input_node[i])
                    self.input_node[i].child.append(node_)
            if self.with_skip_connection and layer_id > 0:
                skip_con = np.where(
                    self.arc_seq[arc_pointer + 1 + len(self.input_node) * self.with_input_blocks: arc_pointer + 1 + len(
                        self.input_node) * self.with_input_blocks + layer_id] == 1)[0]
                # print(layer_id, skip_con)
                for i in skip_con:
                    parent.append(nodes[i])
                    nodes[i].child.append(node_)
            else:
                if layer_id > 0:
                    parent.append(nodes[-1])
                    nodes[-1].child.append(node_)
                # leave first layer without parent, so it
                # will be connected to default input node
            node_.parent = parent
            nodes.append(node_)
            arc_pointer += 1 + int(self.with_input_blocks) * len(self.input_node) + int(
                self.with_skip_connection) * layer_id
        # print('initial', nodes)
        self._aux_loss(nodes)
        return nodes


class EnasAnnDAG:
    def __init__(self,
                 model_space,
                 input_node,
                 output_node,
                 model_compile_dict,
                 session,
                 l1_reg=0.0,
                 l2_reg=0.0,
                 with_skip_connection=True,
                 with_input_blocks=True,
                 with_output_blocks=False,
                 controller=None,
                 feature_model=None,
                 feature_model_trainable=None,
                 child_train_op_kwargs=None,
                 name='EnasDAG'):
        """
        EnasAnnDAG is a DAG model builder for using the weight sharing framework. This class deals with the vanilla
        Artificial neural network. The weight sharing is between all Ws for different hidden units sizes - that is,
        a larger hidden size always includes the smaller ones.
        Args:
            model_space:
            input_node:
            output_node:
            model_compile_dict: compile dict for child models
            session: tf.Session
            with_skip_connection:
            with_input_blocks:
            name:
        """
        #assert with_skip_connection == with_input_blocks == True, \
        #    "EnasAnnDAG must have with_input_blocks and with_skip_connection"
        self.model_space = model_space
        if not type(input_node) in (tuple, list):
            self.input_node = [input_node]
        else:
            self.input_node = input_node
        if not type(output_node) in (tuple, list):
            self.output_node = [output_node]
        else:
            self.output_node = output_node
        if session is None:
            self.session = tf.compat.v1.Session()
        else:
            self.session = session
        self.model_compile_dict = model_compile_dict
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.with_skip_connection = with_skip_connection
        self.with_input_blocks = with_input_blocks
        self.with_output_blocks = with_output_blocks
        self.num_layers = len(model_space)
        self.name = name
        self.input_arc = None
        self.sample_w_masks = []
        self.feature_model = feature_model
        if self.feature_model is None:
            self.feature_model_trainable = False
        else:
            self.feature_model_trainable = feature_model_trainable or False
        self.child_train_op_kwargs = child_train_op_kwargs

        self._verify_args()
        self._create_params()
        if controller is None:
            self.controller = None
        else:
            self.controller = controller
            self._build_sample_arc()
        self._build_fixed_arc()
        vars = [v for v in tf.all_variables() if v.name.startswith(self.name)]
        self.session.run(tf.initialize_variables(vars))

        # for compatability with EnasConv1d
        self.train_fixed_arc = False

    def __call__(self, arc_seq, node_builder=None, *args, **kwargs):
        model = self._model(arc_seq)
        if node_builder is not None and arc_seq is not None:
            nb = node_builder(arc_seq)
            nodes = nb._init_nodes()
            nodes = nb._prune_nodes(nodes)
            nodes = nb.input_node + nodes + [nb.output_node]
            model.nodes = nodes
        return model

    def set_controller(self, controller):
        assert self.controller is None, "already has inherent controller, disallowed; start a new " \
                                        "EnasAnnDAG instance if you want to connect another controller"
        self.controller = controller
        self._build_sample_arc()
        vars = [v for v in tf.all_variables() if v.name.startswith("%s/sample" % self.name)]
        self.session.run(tf.initialize_variables(vars))

    def _verify_args(self):
        """verify vanilla ANN model space, input nodes, etc.,
         and configure internal attr. like masking steps"""
        # check the consistency of with_output_blocks and output_op
        if not self.with_output_blocks and len(self.output_node)>1:
            warnings.warn("You specified `with_output_blocks=False`, but gave a List of output operations of length %i"%len(self.output_node), stacklevel=2)
        # model space
        assert len(set([tuple(self.model_space[i]) for i in
                        range(self.num_layers)])) == 1, "model_space for EnasDAG must be identical for all layers"
        layer_ = self.model_space[0]
        all_actv_fns = set([x.Layer_attributes['activation'] for x in layer_])
        assert len(all_actv_fns) == 1, "all operations must share the same activation function, got %s" % all_actv_fns
        self._actv_fn = all_actv_fns.pop()
        # assert self._actv_fn.lower() == "relu", "EnasDAG only supports ReLU activation function"
        self._weight_units = np.array([x.Layer_attributes['units'] for x in layer_], dtype=np.int32)
        self._weight_max_units = np.max(self._weight_units)

        # input nodes
        # _input_block_map: mapping from input_blocks indicators to feature indices
        self._input_block_map = np.zeros((len(self.input_node), 2), dtype=np.int32)  # n, [start, end]
        self.num_input_blocks = len(self.input_node)
        start_idx = 0
        for i in range(len(self.input_node)):
            n_feature = self.input_node[i].Layer_attributes['shape'][0]
            self._input_block_map[i] = [start_idx, start_idx + n_feature]
            start_idx += n_feature
        self._feature_max_size = start_idx

        # output node
        self._child_output_size = [n.Layer_attributes['units'] for n in self.output_node]
        self._child_output_func = [n.Layer_attributes['activation'] for n in self.output_node]
        self.num_output_blocks = len(self.output_node)
        self._output_block_map = np.array([
            [i * self._weight_max_units, (i + 1) * self._weight_max_units] for i in range(self.num_layers)],
            dtype=np.int32).reshape((self.num_layers, 2))

        # skip connections
        # _skip_conn_map: mapping from skip connection indicators to layer output indices
        self._skip_conn_map = {}
        start_map = np.array([[0, self._weight_max_units]], dtype=np.int32).reshape((1, 2))
        for i in range(1, self.num_layers):
            self._skip_conn_map[i] = start_map  # n, [start, end]
            start_map = np.concatenate([start_map,
                                        np.array([[i * self._weight_max_units, (i + 1) * self._weight_max_units]],
                                                 dtype=np.int32).reshape(1, 2)])

    def _create_params(self):
        self.w = []
        self.b = []
        input_max_size = self._input_block_map[-1][-1]
        with tf.compat.v1.variable_scope(self.name):
            self.train_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="train_step")
            for layer_id in range(self.num_layers):
                with tf.variable_scope("layer_{}".format(layer_id)):
                    self.w.append(tf.compat.v1.get_variable("Weight/w", shape=(
                    input_max_size + layer_id * self._weight_max_units, self._weight_max_units),
                                                            dtype=tf.float32,
                                                            initializer=tf.contrib.layers.variance_scaling_initializer()))
                    self.b.append(tf.compat.v1.get_variable("Bias/b", shape=(self._weight_max_units,),
                                                            initializer=tf.initializers.zeros(),
                                                            dtype=tf.float32))
            with tf.compat.v1.variable_scope("stem_io"):
                self.child_model_input = tf.compat.v1.placeholder(shape=(None, self._feature_max_size),
                                                                  dtype=tf.float32,
                                                                  name="child_input")
                self.child_model_label = [tf.compat.v1.placeholder(shape=(None, self._child_output_size[i]),
                                                                   dtype=tf.float32,
                                                                   name="child_output_%i" % i)
                                          for i in range(len(self.output_node))]
                if self.feature_model is not None:
                    # data pipeline by Tensorflow.data.Dataset.Iterator
                    self.child_model_label_pipe = self.feature_model.y_it
                    self.child_model_input_pipe = self.feature_model.x_it
                # if also learn the connection of output_blocks, need to enlarge the output to allow
                # potential multi-inputs from different hidden layers
                if self.with_output_blocks:
                    self.w_out = [tf.compat.v1.get_variable("w_out_%i" % i, shape=(
                    self._weight_max_units * self.num_layers, self._child_output_size[i]), dtype=tf.float32,
                                                            initializer=tf.contrib.layers.variance_scaling_initializer())
                                  for i in range(len(self.output_node))]
                    self.b_out = [
                        tf.compat.v1.get_variable("b_out_%i" % i, shape=(self._child_output_size[i]), dtype=tf.float32,
                                                  initializer=tf.initializers.zeros())
                        for i in range(len(self.output_node))]
                # otherwise, only need to connect to the last hidden layer
                else:
                    self.w_out = [tf.compat.v1.get_variable("w_out_%i" % i,
                                                            shape=(self._weight_max_units, self._child_output_size[i]),
                                                            dtype=tf.float32,
                                                            initializer=tf.contrib.layers.variance_scaling_initializer())
                                  for i in range(len(self.output_node))]
                    self.b_out = [
                        tf.compat.v1.get_variable("b_out_%i" % i, shape=(self._child_output_size[i]), dtype=tf.float32,
                                                  initializer=tf.initializers.zeros())
                        for i in range(len(self.output_node))]

    def _build_sample_arc(self):
        """
        sample_output and sample_w_masks are the child model tensors that got built after getting a random sample
        from controller.sample_arc
        """
        with tf.compat.v1.variable_scope("%s/sample" % self.name):
            self.connect_controller(self.controller)
            sample_output, sample_w_masks, sample_layer_outputs, sample_dropouts = self._build_dag(self.sample_arc)
            self.sample_model_output = sample_output
            self.sample_w_masks = sample_w_masks
            self.sample_layer_outputs = sample_layer_outputs
            ops = self._compile(w_masks=self.sample_w_masks, model_output=self.sample_model_output,
                                use_pipe=True,
                                # var_scope="%s/sample" % self.name
                                )
            self.sample_train_op = ops['train_op']
            self.sample_loss = ops['loss']
            self.sample_optimizer = ops['optimizer']
            self.sample_metrics = ops['metrics']
            self.sample_ops = ops
            self.sample_dropouts = sample_dropouts

    def _build_fixed_arc(self):
        """
        fixed_output and fixed_w_masks are the child model tensors built according to a fixed arc from user inputs
        """
        with tf.compat.v1.variable_scope("%s/fixed" % self.name):
            self._create_input_ph()
            fixed_output, fixed_w_masks, fixed_layer_outputs, fixed_dropouts = self._build_dag(self.input_arc)
            self.fixed_model_output = fixed_output
            self.fixed_w_masks = fixed_w_masks
            self.fixed_layer_outputs = fixed_layer_outputs
            ops = self._compile(w_masks=self.fixed_w_masks, model_output=self.fixed_model_output,
                                use_pipe=False,
                                # var_scope = "%s/fixed" % self.name
                                )
            self.fixed_train_op = ops['train_op']
            self.fixed_loss = ops['loss']
            self.fixed_optimizer = ops['optimizer']
            self.fixed_metrics = ops['metrics']
            self.fixed_ops = ops
            self.fixed_dropouts = fixed_dropouts

    def _build_dag(self, arc_seq):
        """
        Shared DAG building process for both sampled arc and fixed arc
        Args:
            arc_seq:

        Returns:

        """
        w_masks = []
        layer_outputs = []
        start_idx = 0
        dropout_placeholders = []
        # if no addition of feature model, use dataset directly
        if self.feature_model is None:
            inputs = self.child_model_input
        # otherwise, need to connect to feature model output
        else:
            # TODO: for now, only use data_pipe for sample_arc
            if self.feature_model.pseudo_inputs_pipe is None or type(arc_seq) is list:
                # print('='*80)
                # print('used placeholder')
                inputs = self.feature_model.pseudo_inputs
            else:
                # print('='*80)
                # print('used data pipe')
                inputs = self.feature_model.pseudo_inputs_pipe
        input_dropprob = tf.placeholder_with_default(0.0, shape=(), name='dropout_input')
        inputs = tf.nn.dropout(inputs, rate=input_dropprob)
        dropout_placeholders.append(input_dropprob)
        for layer_id in range(self.num_layers):
            w = self.w[layer_id]
            b = self.b[layer_id]
            # column masking for output units
            num_units = tf.nn.embedding_lookup(self._weight_units, arc_seq[start_idx])
            col_mask = tf.cast(tf.less(tf.range(0, limit=self._weight_max_units, delta=1), num_units), tf.int32)
            start_idx += 1

            # input masking for with_input_blocks
            if self.with_input_blocks:
                inp_mask = arc_seq[start_idx: start_idx + self.num_input_blocks]
                inp_mask = tf.boolean_mask(self._input_block_map, tf.squeeze(inp_mask))
                new_range = tf.range(0, limit=self._feature_max_size, dtype=tf.int32)
                inp_mask = tf.map_fn(lambda x: tf.cast(tf.logical_and(x[0] <= new_range, new_range < x[1]), dtype=tf.int32),
                                     inp_mask)
                inp_mask = tf.reduce_sum(inp_mask, axis=0)
                start_idx += self.num_input_blocks * self.with_input_blocks
            else:
                # get all inputs if layer_id=0, else mask all
                inp_mask = tf.ones(shape=(self._feature_max_size), dtype=tf.int32) if layer_id == 0 else \
                           tf.zeros(shape=(self._feature_max_size), dtype=tf.int32)

            # hidden layer masking for with_skip_connection
            if self.with_skip_connection:
                if layer_id > 0:
                    layer_mask = arc_seq[
                                 start_idx: start_idx + layer_id]
                    layer_mask = tf.boolean_mask(self._skip_conn_map[layer_id], layer_mask)
                    new_range2 = tf.range(0, limit=layer_id * self._weight_max_units, delta=1, dtype=tf.int32)
                    layer_mask = tf.map_fn(
                        lambda t: tf.cast(tf.logical_and(t[0] <= new_range2, new_range2 < t[1]), dtype=tf.int32),
                        layer_mask)
                    layer_mask = tf.reduce_sum(layer_mask, axis=0)
                    start_idx += layer_id * self.with_skip_connection
                else:
                    layer_mask = []
                row_mask = tf.concat([inp_mask, layer_mask], axis=0)
            else:
                if layer_id > 0:
                    # keep last/closest layer, mask all others
                    layer_masks = [tf.zeros(shape=(self._weight_max_units*(layer_id-1)), dtype=tf.int32), tf.ones(shape=(self._weight_max_units), dtype=tf.int32)]
                else:
                    layer_masks = []

                row_mask = tf.concat([inp_mask] + layer_masks, axis=0)
            w_mask = tf.matmul(tf.expand_dims(row_mask, -1), tf.expand_dims(col_mask, 0))

            # get the TF layer
            w = tf.where(tf.cast(w_mask, tf.bool), x=w, y=tf.fill(tf.shape(w), 0.))
            b = tf.where(tf.cast(col_mask, tf.bool), x=b, y=tf.fill(tf.shape(b), 0.))
            x, drop_rate = self._layer(w, b, inputs, layer_id)
            dropout_placeholders.append(drop_rate)
            layer_outputs.append(x)
            inputs = tf.concat([inputs, x], axis=1)
            w_masks.append((w, b))

        if self.with_output_blocks:
            model_output = []
            output_arcs = arc_seq[start_idx::]
            if type(output_arcs) is list:
                output_arcs_len = len(output_arcs)
            else:  # is tensor
                output_arcs_len = output_arcs.shape[0].value
            assert output_arcs_len == self.num_output_blocks * self.num_layers, "model builder was specified to build output" \
                                                                                " connections, but the input architecture did" \
                                                                                "n't match output info; expected arc of length" \
                                                                                "=%i, received %i" % (
                                                                                start_idx + self.num_output_blocks * self.num_layers,
                                                                                len(arc_seq) if type(
                                                                                    arc_seq) is list else arc_seq.shape[
                                                                                    0].value)
            layer_outputs_ = tf.concat(layer_outputs, axis=1)  # shape: num_samples, max_units x num_layers
            for i in range(self.num_output_blocks):
                # output_mask is a row_mask
                output_mask = tf.boolean_mask(self._output_block_map,
                                              output_arcs[i * self.num_layers: (i + 1) * self.num_layers])
                new_range = tf.range(0, limit=self.num_layers * self._weight_max_units, delta=1, dtype=tf.int32)
                output_mask = tf.map_fn(
                    lambda t: tf.cast(tf.logical_and(t[0] <= new_range, new_range < t[1]), dtype=tf.int32),
                    output_mask)
                output_mask = tf.reduce_sum(output_mask, axis=0)
                output_mask = tf.matmul(tf.expand_dims(output_mask, -1),
                                        tf.ones((1, self._child_output_size[i]), dtype=tf.int32))
                w = tf.where(tf.cast(output_mask, tf.bool), x=self.w_out[i], y=tf.fill(tf.shape(self.w_out[i]), 0.))
                model_output.append(
                    get_tf_layer(self._child_output_func[i])(tf.matmul(layer_outputs_, w) + self.b_out[i]))
                w_masks.append((w, self.b_out[i]))
        else:
            model_output = [get_tf_layer(self._child_output_func[i])(tf.matmul(x, self.w_out[i]) + self.b_out[i])
                            for i in range(len(self.output_node))]
        return model_output, w_masks, layer_outputs, dropout_placeholders

    def _layer(self, w, b, inputs, layer_id, use_dropout=True):
        layer = get_tf_layer(self._actv_fn)(tf.matmul(inputs, w) + b)
        if use_dropout:
            drop_prob = tf.placeholder_with_default(0.0, shape=(), name='dropout_%i' % layer_id)
            layer = tf.nn.dropout(layer, rate=drop_prob)
            return layer, drop_prob
        else:
            return layer, None

    def _model(self, arc):
        if self.feature_model is None:
            child_model_input = self.child_model_input
        else:
            if self.feature_model.pseudo_inputs_pipe is None or arc is not None:
                child_model_input = self.feature_model.x_inputs
            else:
                child_model_input = self.child_model_input_pipe
        if arc is None:
            model = EnasAnnModel(inputs=child_model_input, outputs=self.sample_model_output,
                                 arc_seq=arc,
                                 dag=self,
                                 session=self.session)
        else:
            model = EnasAnnModel(inputs=child_model_input, outputs=self.fixed_model_output,
                                 arc_seq=arc,
                                 dag=self,
                                 session=self.session)
        return model

    def _create_input_ph(self):
        ops_each_layer = 1
        total_arc_len = sum([ops_each_layer + self._input_block_map.shape[0] * self.with_input_blocks] + [
            ops_each_layer + self._input_block_map.shape[0] * self.with_input_blocks + i * self.with_skip_connection
            for i in range(1, self.num_layers)])
        if self.with_output_blocks:
            total_arc_len += self.num_output_blocks * self.num_layers
        self.input_ph_ = [tf.compat.v1.placeholder(shape=(), dtype=tf.int32, name='arc_{}'.format(i))
                          for i in range(total_arc_len)]
        self.input_arc = self.input_ph_
        return

    def _compile(self, w_masks, model_output, use_pipe=True, var_scope=None):
        """
        Compile loss and train_op here so all child models will share the same, instead of creating new ones every time
        """
        loss = self.model_compile_dict['loss']
        optimizer = self.model_compile_dict['optimizer']
        metrics = self.model_compile_dict['metrics'] if 'metrics' in self.model_compile_dict else None
        var_scope = var_scope or self.name

        with tf.compat.v1.variable_scope("compile", reuse=tf.AUTO_REUSE):
            if self.feature_model is None or self.feature_model.pseudo_inputs_pipe is None:
                labels = self.child_model_label
            else:
                # TODO: for now, only use data_pipe for sample_arc
                if use_pipe:
                    labels = self.child_model_label_pipe
                else:
                    labels = self.child_model_label

            # TODO: process loss_weights
            loss_weights = self.model_compile_dict['loss_weights'] if 'loss_weights' in self.model_compile_dict else None
            if type(loss) is str:
                loss_ = [get_tf_loss(loss, labels[i], model_output[i]) for i in range(len(model_output))]
                # loss_ should originally all have the batch_size dim, then reduce_mean to 1-unit sample
                # if the loss is homogeneous, then take the mean
                loss_ = tf.reduce_mean(loss_)
            elif type(loss) is list:
                loss_ = []
                for i in range(len(loss)):
                    loss_.append(get_tf_loss(
                        loss[i],
                        labels[i],
                        model_output[i]
                    ))
                # if the loss are provided by a list, assume its heterogeneous, and return the sum of
                # each individual loss components
                loss_ = tf.reduce_sum(loss_)
            elif callable(loss):
                loss_ = tf.reduce_sum([loss(labels[i], model_output[i]) for i in range(len(model_output))])
            else:
                raise Exception("expect loss to be str, list, dict or callable; got %s" % loss)
            trainable_var = [var for var in tf.trainable_variables() if var.name.startswith(var_scope)]
            if self.feature_model_trainable:
                feature_model_trainable_var = [var for var in tf.trainable_variables() if
                                               var.name.startswith(self.feature_model.name)]
                assert len(feature_model_trainable_var) > 0, "You asked me to train featureModel but there is no trainable " \
                                                             "variables in featureModel"
                trainable_var += feature_model_trainable_var
            regularization_penalty = 0.
            if self.l1_reg > 0:
                l1_regularizer = tf.contrib.layers.l1_regularizer(
                    scale=self.l1_reg, scope=self.name
                )
                l1_regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer,
                                                                                   [var[0] for var in w_masks])
                loss_ += l1_regularization_penalty
            else:
                l1_regularization_penalty = 0.

            if self.l2_reg > 0:
                l2_regularizer = tf.contrib.layers.l2_regularizer(
                    scale=self.l2_reg, scope=self.name
                )
                l2_regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer,
                                                                                   [var[0] for var in w_masks])
                loss_ += l2_regularization_penalty
            else:
                l2_regularization_penalty = 0.

            regularization_penalty += l1_regularization_penalty + l2_regularization_penalty

            # default settings used from enas
            if self.child_train_op_kwargs is None:
                # more sensible default values
                train_op, lr, grad_norm, optimizer_ = get_keras_train_ops(
                    # get_train_ops(
                    loss=loss_,
                    tf_variables=trainable_var,
                    optim_algo=optimizer,
                    train_step=self.train_step)
            # user specific settings; useful when training the final searched arc
            else:
                train_op, lr, grad_norm, optimizer_ = get_keras_train_ops(
                    # get_train_ops
                    loss=loss_,
                    tf_variables=trainable_var,
                    train_step=self.train_step,
                    optim_algo=optimizer,
                    **self.child_train_op_kwargs)
            if metrics is None:
                metrics = []
            else:
                metrics = [get_tf_metrics(x) for x in metrics]
            # TODO: this needs fixing to be more generic;
            # TODO: for example, the squeeze op is not usable for
            # TODO: other metrics such as Acc
            metrics_ = [f(tf.squeeze(self.child_model_label[i]), tf.squeeze(model_output[i]))
                        for i in range(len(model_output)) for f in metrics]
            ops = {'train_op': train_op,
                   'lr': lr,
                   'grad_norm': grad_norm,
                   'optimizer': optimizer,
                   'loss': loss_,
                   'metrics': metrics_,
                   'reg_cost': regularization_penalty
                   }
            return ops

    def connect_controller(self, controller):
        self.sample_arc = controller.sample_arc
        self.controller = controller
        return


class EnasConv1dDAG:
    def __init__(self,
                 model_space,
                 input_node,
                 output_node,
                 model_compile_dict,
                 session,
                 with_skip_connection=True,
                 batch_size=128,
                 keep_prob=0.9,
                 l1_reg=0.0,
                 l2_reg=0.0,
                 reduction_factor=4,
                 controller=None,
                 child_train_op_kwargs=None,
                 stem_config=None,
                 data_format='NWC',
                 train_fixed_arc=False,
                 fixed_arc=None,
                 name='EnasDAG',
                 **kwargs):
        """EnasCnnDAG is a DAG model builder for using the weight sharing framework.

        This class deals with the Convolutional neural network.

        Parameters
        ----------
        model_space: amber.architect.ModelSpace
        input_node: amber.architect.Operation, or list
        output_node: amber.architect.Operation, or list
        model_compile_dict: dict
            compile dict for child models
        session: tf.Session
            session for building enas DAG
        train_fixed_arc: bool
            boolean indicator for whether is the final stage; if is True, must provide `fixed_arc` and not connect
            to a controller
        fixed_arc: list-like
            the architecture for final stage training
        name: str
        """
        assert type(input_node) in (State, tf.Tensor) or len(
            input_node) == 1, "EnasCnnDAG currently does not accept List type of inputs"
        assert type(output_node) in (State, tf.Tensor) or len(
            output_node) == 1, "EnasCnnDAG currently does not accept List type of outputs"
        self.input_node = input_node
        self.output_node = output_node
        self.num_layers = len(model_space)
        self.model_space = model_space
        self.model_compile_dict = model_compile_dict
        self.session = session
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.with_skip_connection = with_skip_connection
        self.controller = None
        if controller is not None:
            self.set_controller(controller)
        self.child_train_op_kwargs = child_train_op_kwargs
        self.stem_config = stem_config or {}

        self.name = name
        self.batch_size = batch_size
        self.batch_dim = None
        self.reduction_factor = reduction_factor
        self.keep_prob = keep_prob
        self.data_format = data_format
        self.out_filters = None
        self.branches = []
        self.is_initialized = False

        self.add_conv1_under_pool = kwargs.pop("add_conv1_under_pool", True)

        self.train_fixed_arc = train_fixed_arc
        self.fixed_arc = fixed_arc
        if self.train_fixed_arc:
            assert self.fixed_arc is not None, "if train_fixed_arc=True, must provide the architectures in `fixed_arc`"
            assert controller is None, "if train_fixed_arc=True, must not provide controller"
            self.skip_max_depth = None

        self._verify_args()
        self.vars = []
        if controller is None:
            self.controller = None
            print("this EnasDAG instance did not connect a controller; pleaes make sure you are only training a fixed "
                  "architecture.")
        else:
            self.controller = controller
            self._build_sample_arc()
        self._build_fixed_arc()
        self.session.run(tf.initialize_variables(self.vars))

    def _verify_args(self):
        out_filters = []
        pool_layers = []
        for layer_id in range(len(self.model_space)):
            layer = self.model_space[layer_id]
            this_out_filters = [l.Layer_attributes['filters'] for l in layer]
            assert len(
                set(this_out_filters)) == 1, "EnasConv1dDAG only supports one identical number of filters per layer," \
                                             "but found %i different number of filters in layer %s" % \
                                             (len(set(this_out_filters)), layer)
            if len(out_filters) and this_out_filters[0] != out_filters[-1]:
                pool_layers.append(layer_id - 1)

            out_filters.append(this_out_filters[0])
        self.out_filters = out_filters
        self.pool_layers = pool_layers

        # if train fixed arc, avoid building unused skip connections
        # and verify the input fixed_arc
        if self.train_fixed_arc:
            assert self.fixed_arc is not None
            skip_max_depth = {}
            start_idx = 0
            for layer_id in range(len(self.model_space)):
                skip_max_depth[layer_id] = layer_id
                operation = self.fixed_arc[start_idx]
                total_choices = len(self.model_space[layer_id])
                assert 0 <= operation < total_choices, "Invalid operation selection: layer_id=%i, " \
                                                       "operation=%i, model space len=%i" % (
                                                       layer_id, operation, total_choices)
                if layer_id > 0:
                    skip_binary = self.fixed_arc[(start_idx + 1):(start_idx + 1 + layer_id)]
                    skip = [i for i in range(layer_id) if skip_binary[i] == 1]
                    for d in skip:
                        skip_max_depth[d] = layer_id

                start_idx += 1 + layer_id
            print('-' * 80)
            print(skip_max_depth)
            self.skip_max_depth = skip_max_depth

        if type(self.input_node) is list:
            self.input_node = self.input_node[0]
        self.input_ph = tf.placeholder(shape=[self.batch_dim] + list(self.input_node.Layer_attributes['shape']),
                                       name='child_input_placeholder',
                                       dtype=tf.float32)
        if type(self.output_node) is list:
            self.output_node = self.output_node[0]
        self.label_ph = tf.placeholder(shape=(self.batch_dim, self.output_node.Layer_attributes['units']),
                                       dtype=tf.float32,
                                       name='child_output_placeholder')

    def __call__(self, arc_seq=None, **kwargs):
        return self._model(arc_seq, **kwargs)

    def _model(self, arc, **kwargs):
        if self.train_fixed_arc:
            assert arc == self.fixed_arc or arc is None, "This DAG instance is built to train fixed arc, hence you " \
                                                         "can only provide arc=None or arc=self.fixed_arc; check the " \
                                                         "initialization of this instances "
        if arc is None:
            if self.train_fixed_arc:
                model = EnasCnnModel(inputs=self.fixed_model_input,
                                     outputs=self.fixed_model_output,
                                     labels=self.fixed_model_label,
                                     arc_seq=arc,
                                     dag=self,
                                     session=self.session,
                                     name=self.name)

            else:
                model = EnasCnnModel(inputs=self.sample_model_input,
                                     outputs=self.sample_model_output,
                                     labels=self.sample_model_label,
                                     arc_seq=arc,
                                     dag=self,
                                     session=self.session,
                                     name=self.name)
        else:
            model = EnasCnnModel(inputs=self.fixed_model_input,
                                 outputs=self.fixed_model_output,
                                 labels=self.fixed_model_label,
                                 arc_seq=arc,
                                 dag=self,
                                 session=self.session,
                                 name=self.name)
        return model

    def set_controller(self, controller):
        assert self.controller is None, "already has inherent controller, disallowed; start a new " \
                                        "EnasCnnDAG instance if you want to connect another controller"
        self.controller = controller
        self.sample_arc = controller.sample_arc

    def _build_sample_arc(self, input_tensor=None, label_tensor=None, **kwargs):
        """
        Notes:
            I left `input_tensor` and `label_tensor` so that in the future some pipeline
            tensors can be connected to the model, instead of the placeholders as is now.

        Args:
            input_tensor:
            label_tensor:
            **kwargs:

        Returns:

        """
        var_scope = self.name
        is_training = kwargs.pop('is_training', True)
        reuse = kwargs.pop('reuse', tf.AUTO_REUSE)
        with tf.compat.v1.variable_scope(var_scope, reuse=reuse):
            input_tensor = self.input_ph if input_tensor is None else input_tensor
            label_tensor = self.label_ph if label_tensor is None else label_tensor
            model_out, dropout_placeholders = self._build_dag(arc_seq=self.sample_arc, input_tensor=input_tensor,
                                                              is_training=is_training,
                                                              reuse=reuse)
            self.sample_model_output = model_out
            self.sample_model_input = input_tensor
            self.sample_model_label = label_tensor
            self.sample_dropouts = dropout_placeholders
            ops = self._compile(model_output=[self.sample_model_output], labels=[label_tensor],
                                is_training=is_training,
                                var_scope=var_scope)
            self.sample_train_op = ops['train_op']
            self.sample_loss = ops['loss']
            self.sample_optimizer = ops['optimizer']
            self.sample_metrics = ops['metrics']
            self.sample_ops = ops
        # if not self.is_initialized:
        vars_ = [v for v in tf.all_variables() if v.name.startswith(var_scope) and v not in self.vars]
        if len(vars_):
            self.session.run(tf.initialize_variables(vars_))
            self.vars += vars_
            # self.is_initialized = True

    def _build_fixed_arc(self, input_tensor=None, label_tensor=None, **kwargs):
        """
        Notes:
            I left `input_tensor` and `label_tensor` so that in the future some pipeline
            tensors can be connected to the model, instead of the placeholders as is now.

        Args:
            input_tensor:
            label_tensor:
            **kwargs:

        Returns:

        """
        var_scope = self.name
        if self.train_fixed_arc:
            is_training = True
        else:
            is_training = kwargs.pop('is_training', False)
        reuse = kwargs.pop('reuse', tf.AUTO_REUSE)
        with tf.compat.v1.variable_scope(var_scope, reuse=reuse):
            input_tensor = self.input_ph if input_tensor is None else input_tensor
            label_tensor = self.label_ph if label_tensor is None else label_tensor
            self._create_input_ph()
            if self.train_fixed_arc:
                model_out, dropout_placeholders = self._build_dag(arc_seq=self.fixed_arc, input_tensor=input_tensor,
                                                                  is_training=is_training,
                                                                  reuse=reuse)
            else:
                model_out, dropout_placeholders = self._build_dag(arc_seq=self.input_arc, input_tensor=input_tensor,
                                                                  is_training=is_training,
                                                                  reuse=reuse)
            self.fixed_model_output = model_out
            self.fixed_model_input = input_tensor
            self.fixed_model_label = label_tensor
            self.fixed_dropouts = dropout_placeholders
            ops = self._compile(model_output=[self.fixed_model_output], labels=[label_tensor],
                                is_training=is_training,
                                var_scope=var_scope)
            self.fixed_train_op = ops['train_op']
            self.fixed_loss = ops['loss']
            self.fixed_optimizer = ops['optimizer']
            self.fixed_metrics = ops['metrics']
            self.fixed_ops = ops
        # if not self.is_initialized:
        vars = [v for v in tf.all_variables() if v.name.startswith(var_scope) and v not in self.vars]
        if len(vars):
            self.session.run(tf.initialize_variables(vars))
            self.vars += vars
            # self.is_initialized = True

    def _create_input_ph(self):
        ops_each_layer = 1
        total_arc_len = sum([ops_each_layer + i
                             for i in range(self.num_layers)])
        input_ph_ = [tf.compat.v1.placeholder(shape=(), dtype=tf.int32, name='arc_{}'.format(i))
                     for i in range(total_arc_len)]
        self.input_arc = input_ph_
        return

    def _compile(self, model_output, labels=None, is_training=True, var_scope=None):
        loss = self.model_compile_dict['loss']
        optimizer = self.model_compile_dict['optimizer']
        metrics = self.model_compile_dict.pop('metrics', None)
        var_scope = var_scope or self.name
        labels = self.label_ph if labels is None else labels
        with tf.variable_scope('compile'):
            if type(loss) is str:
                loss_ = [get_tf_loss(loss, labels[i], model_output[i]) for i in range(len(model_output))]
                # loss_ should originally all have the batch_size dim, then reduce_mean to 1-unit sample
                # if the loss is homogeneous, then take the mean
                loss_ = tf.reduce_mean(loss_)
            elif type(loss) is list:
                loss_ = []
                for i in range(len(loss)):
                    loss_.append(get_tf_loss(
                        loss[i],
                        labels[i],
                        model_output[i]
                    ))
                # if the loss are provided by a list, assume its heterogeneous, and return the sum of
                # each individual loss components
                loss_ = tf.reduce_sum(loss_)
            elif callable(loss):
                loss_ = tf.reduce_sum([loss(labels[i], model_output[i]) for i in range(len(model_output))])
            else:
                raise Exception("expect loss to be str, list, dict or callable; got %s" % loss)
            trainable_var = [var for var in tf.trainable_variables() if var.name.startswith(var_scope)]
            regularization_penalty = 0.
            if self.l1_reg > 0:
                l1_regularizer = tf.contrib.layers.l1_regularizer(
                    scale=self.l1_reg, scope=self.name
                )
                l1_regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer,
                                                                                   [var for var in trainable_var if
                                                                                    var.name.split('/')[-1] == 'w:0'])
                loss_ += l1_regularization_penalty
            else:
                l1_regularization_penalty = 0.

            if self.l2_reg > 0:
                l2_regularizer = tf.contrib.layers.l2_regularizer(
                    scale=self.l2_reg, scope=self.name
                )
                l2_regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer,
                                                                                   [var for var in trainable_var if
                                                                                    var.name.split('/')[-1] == 'w:0'])
                loss_ += l2_regularization_penalty
            else:
                l2_regularization_penalty = 0.

            regularization_penalty += l1_regularization_penalty + l2_regularization_penalty

            if is_training:
                # default settings used from enas
                if self.child_train_op_kwargs is None:
                    # more sensible default values
                    train_op, lr, grad_norm, optimizer_ = get_keras_train_ops(  # get_train_ops(
                        loss=loss_,
                        tf_variables=trainable_var,
                        optim_algo=optimizer,
                        train_step=self.train_step)
                # user specific settings; useful when training the final searched arc
                else:
                    train_op, lr, grad_norm, optimizer_ = get_keras_train_ops(  # get_train_ops(
                        loss=loss_,
                        tf_variables=trainable_var,
                        train_step=self.train_step,
                        optim_algo=optimizer,
                        **self.child_train_op_kwargs)
            else:
                train_op, lr, grad_norm, optimizer_ = None, None, None, None
            if metrics is None:
                metrics = []
            else:
                metrics = [get_tf_metrics(x) for x in metrics]
            # TODO: this needs fixing to be more generic;
            # TODO: for example, the squeeze op is not usable for
            # TODO: other metrics such as Acc
            metrics_ = [f(labels[i], model_output[i])
                        for i in range(len(model_output)) for f in metrics]
            ops = {'train_op': train_op,
                   'lr': lr,
                   'grad_norm': grad_norm,
                   'optimizer': optimizer_,
                   'loss': loss_,
                   'metrics': metrics_,
                   'reg_cost': regularization_penalty
                   }
        return ops

    def _build_dag(self, arc_seq, input_tensor=None, is_training=True, reuse=True):
        self.layers = []
        self.train_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="train_step")
        dropout_placeholders = []
        with tf.variable_scope('child_model', reuse=reuse):
            out_filters = self.out_filters
            # input = self.input_node
            input = self.input_ph if input_tensor is None else input_tensor
            layers = []
            #has_stem_conv = self.stem_config['has_stem_conv'] if 'has_stem_conv' in self.stem_config else True
            has_stem_conv = True
            if has_stem_conv:
                with tf.variable_scope("stem_conv"):
                    stem_kernel_size = self.stem_config.pop('stem_kernel_size', 8)
                    stem_filters = out_filters[0]
                    #w = create_weight("w", [stem_kernel_size, 4, stem_filters])
                    w = create_weight("w", [stem_kernel_size, 1, stem_filters])
                    x = tf.nn.conv1d(input, w, 1, "SAME", data_format=self.data_format)
                    x = batch_norm1d(x, is_training, data_format=self.data_format)
                    layers.append(x)
                    self.layers.append(x)
            else:
                layers = [input]

            start_idx = 0
            for layer_id in range(self.num_layers):
                with tf.variable_scope("layer_{0}".format(layer_id)):
                    x = self._layer(arc_seq, layer_id, layers, start_idx, out_filters[layer_id], is_training)
                    if is_training:
                        dropout_placeholders.append(
                            tf.placeholder_with_default(1 - self.keep_prob, shape=(), name="dropout_%s" % layer_id)
                        )
                        x = tf.nn.dropout(x, rate=dropout_placeholders[-1])
                    if layer_id == 0:
                        layers = [x]
                    else:
                        layers.append(x)
                    self.layers.append(x)
                    if (self.with_skip_connection is True) and (layer_id in self.pool_layers):
                        with tf.variable_scope("pool_at_{0}".format(layer_id)):
                            pooled_layers = []
                            for i, layer in enumerate(layers):
                                if self.train_fixed_arc and self.skip_max_depth[i] < layer_id:
                                    print("Not building pool_at_%i/from_%i because its max-depth=%i" % (
                                    layer_id, i, self.skip_max_depth[i]))
                                    x = layer
                                else:
                                    with tf.variable_scope("from_{0}".format(i)):
                                        x = self._refactorized_channels_for_skipcon(
                                            layer, out_filters[layer_id + 1], is_training)
                                pooled_layers.append(x)
                            layers = pooled_layers
                start_idx += 1 + layer_id*int(self.with_skip_connection)

            flatten_op = self.stem_config['flatten_op'] if 'flatten_op' in self.stem_config else 'flatten'
            if flatten_op == 'global_avg_pool' or flatten_op == 'gap':
                keras_data_format = 'channels_last' if self.data_format.endswith('C') else "channels_first"
                x = tf.keras.layers.GlobalAveragePooling1D(data_format=keras_data_format)(x)
            elif flatten_op == 'flatten':
                keras_data_format = 'channels_last' if self.data_format.endswith('C') else "channels_first"
                x = tf.keras.layers.Flatten(data_format=keras_data_format)(x)
            else:
                raise Exception("cannot understand flatten_op: %s" % flatten_op)
            self.layers.append(x)

            if is_training:
                dropout_placeholders.append(
                    tf.placeholder_with_default(1 - self.keep_prob, shape=(), name="last_conv_dropout")
                )
                x = tf.nn.dropout(x, rate=dropout_placeholders[-1])
            with tf.variable_scope("fc"):
                fc_units = self.stem_config['fc_units'] if 'fc_units' in self.stem_config else 1000
                if flatten_op == 'global_avg_pool' or flatten_op == 'gap':
                    try:
                        inp_c = x.get_shape()[-1].value
                    except AttributeError:
                        inp_c = x.get_shape()[-1]
                    w = create_weight("w_fc", [inp_c, fc_units])
                elif flatten_op == 'flatten':
                    try:
                        inp_c = np.prod(x.get_shape()[1:]).value
                    except AttributeError:
                        inp_c = np.prod(x.get_shape()[1:])
                    w = create_weight("w_fc", [inp_c, fc_units])
                else:
                    raise Exception("Unknown fc string: %s" % flatten_op)
                b = create_bias("b_fc", shape=[fc_units])
                x = tf.matmul(x, w) + b
                x = tf.nn.relu(x)
                if is_training:
                    dropout_placeholders.append(
                        tf.placeholder_with_default(1 - self.keep_prob, shape=(), name="last_conv_dropout")
                    )
                    x = tf.nn.dropout(x, rate=dropout_placeholders[-1])

                w_out = create_weight("w_out", [fc_units, self.output_node.Layer_attributes['units']])
                b_out = create_bias("b_out", shape=[self.output_node.Layer_attributes['units']])
                model_output = get_tf_layer(self.output_node.Layer_attributes['activation'])(
                    tf.matmul(x, w_out) + b_out)
        return model_output, dropout_placeholders

    def _refactorized_channels_for_skipcon(self, layer, out_filters, is_training):
        """for dealing with mismatch-dimensions in skip connections: use a linear transformation"""
        if self.data_format == 'NWC':
            try:
                inp_c = layer.get_shape()[-1].value
            except AttributeError:
                inp_c = layer.get_shape()[-1]
            actual_data_format = 'channels_last'
        elif self.data_format == 'NCW':
            try:    
                inp_c = layer.get_shape()[1].value
            except AttributeError:
                inp_c = layer.get_shape()[1]
            actual_data_format = 'channels_first'

        with tf.variable_scope("path1_conv"):
            w = create_weight("w", [1, inp_c, out_filters])
            x = tf.nn.conv1d(layer, filters=w, stride=1, padding="SAME")
            x = tf.layers.max_pooling1d(
                x, self.reduction_factor, self.reduction_factor, "SAME", data_format=actual_data_format)
        return x

    def _layer(self, arc_seq, layer_id, prev_layers, start_idx, out_filters, is_training):
        inputs = prev_layers[-1]
        if self.data_format == "NWC":
            try:
                inp_w = inputs.get_shape()[1].value
                inp_c = inputs.get_shape()[2].value
            except AttributeError:    # for newer tf2
                inp_w = inputs.get_shape()[1]
                inp_c = inputs.get_shape()[2]

        elif self.data_format == "NCW":
            try:
                inp_c = inputs.get_shape()[1].value
                inp_w = inputs.get_shape()[2].value
            except AttributeError:
                inp_c = inputs.get_shape()[1]
                inp_w = inputs.get_shape()[2]

        else:
            raise Exception("cannot understand data format: %s" % self.data_format)
        count = arc_seq[start_idx]
        branches = {}
        strides = []
        for i in range(len(self.model_space[layer_id])):
            if self.train_fixed_arc and i != count:
                continue

            with tf.variable_scope("branch_%i" % i):
                if self.model_space[layer_id][i].Layer_type == 'conv1d':
                    # print('%i, conv1d'%layer_id)
                    y = self._conv_branch(inputs, layer_attr=self.model_space[layer_id][i].Layer_attributes,
                                          is_training=is_training)
                    branches[tf.equal(count, i)] = y
                elif self.model_space[layer_id][i].Layer_type == 'maxpool1d':
                    # print('%i, maxpool1d' % layer_id)
                    y = self._pool_branch(inputs, "max",
                                          layer_attr=self.model_space[layer_id][i].Layer_attributes,
                                          is_training=is_training)
                    branches[tf.equal(count, i)] = y
                    strides.append(self.model_space[layer_id][i].Layer_attributes['strides'])
                elif self.model_space[layer_id][i].Layer_type == 'avgpool1d':
                    # print('%i, avgpool1d' % layer_id)
                    y = self._pool_branch(inputs, "avg",
                                          layer_attr=self.model_space[layer_id][i].Layer_attributes,
                                          is_training=is_training)
                    branches[tf.equal(count, i)] = y
                    strides.append(self.model_space[layer_id][i].Layer_attributes['strides'])
                elif self.model_space[layer_id][i].Layer_type == 'identity':
                    y = self._identity_branch(inputs)
                    branches[tf.equal(count, i)] = y
                else:
                    raise Exception("Unknown layer: %s" % self.model_space[layer_id][i])

        self.branches.append(branches)
        if len(strides) > 0:
            assert len(set(strides)) == 1, "If you set strides!=1 (i.e. a reduction layer), then all candidate operations must have the same strides to keep the shape identical; got %s" % strides
            inp_w = int(np.ceil(inp_w / strides[0]))
        if self.train_fixed_arc:
            ks = list(branches.keys())
            assert len(ks) == 1
            out = branches[ks[0]]()
        else:
            out = tf.case(
                branches,
                default=lambda: tf.constant(0, tf.float32, shape=[self.batch_size, inp_w, out_filters]),
                exclusive=True)
        if self.data_format == "NWC":
            out.set_shape([None, inp_w, out_filters])
        elif self.data_format == "NCW":
            out.set_shape([None, out_filters, inp_w])
        if self.with_skip_connection is True and layer_id > 0:
            skip_start = start_idx + 1
            skip = arc_seq[skip_start: skip_start + layer_id]
            with tf.variable_scope("skip"):
                # might be the cause of oom.. zz 2020.1.6
                res_layers = []
                for i in range(layer_id):
                    if self.train_fixed_arc:
                        res_layers = [prev_layers[i] for i in range(len(skip)) if skip[i] == 1]
                    else:
                        res_layers.append(tf.cond(tf.equal(skip[i], 1),
                                                  lambda: prev_layers[i],
                                                  lambda: tf.stop_gradient(tf.zeros_like(prev_layers[i]))))
                res_layers.append(out)
                out = tf.add_n(res_layers)

                out = batch_norm1d(out, is_training, data_format=self.data_format)
        return out

    def _conv_branch(self, inputs, layer_attr, is_training):
        kernel_size = layer_attr['kernel_size']
        activation_fn = layer_attr['activation']
        dilation = layer_attr['dilation'] if 'dilation' in layer_attr else 1
        filters = layer_attr['filters']
        if self.data_format == "NWC":
            try:
                inp_c = inputs.get_shape()[-1].value
            except AttributeError:
                inp_c = inputs.get_shape()[-1]
        elif self.data_format == "NCW":
            try:
                inp_c = inputs.get_shape()[1].value
            except AttributeError:
                inp_c = inputs.get_shape()[1]
        w = create_weight("w", [kernel_size, inp_c, filters])
        x = tf.nn.conv1d(inputs, filters=w, stride=1, padding="SAME", dilations=dilation)
        x = batch_norm1d(x, is_training, data_format=self.data_format)
        b = create_bias("b", shape=[1])
        x = get_tf_layer(activation_fn)(x + b)
        return lambda: x

    def _pool_branch(self, inputs, avg_or_max, layer_attr, is_training):
        pool_size = layer_attr['pool_size']
        strides = layer_attr['strides']
        filters = layer_attr['filters']
        if self.data_format == "NWC":
            try:
                inp_c = inputs.get_shape()[-1].value
            except AttributeError:
                inp_c = inputs.get_shape()[-1]
            actual_data_format = "channels_last"
        elif self.data_format == "NCW":
            try:
                inp_c = inputs.get_shape()[1].value
            except AttributeError:
                inp_c = inputs.get_shape()[1]
            actual_data_format = "channels_first"
        else:
            raise Exception("Unknown data format: %s" % self.data_format)

        if self.add_conv1_under_pool:
            with tf.variable_scope("conv_1"):
                w = create_weight("w", [1, inp_c, filters])
                x = tf.nn.conv1d(inputs, w, 1, "SAME", data_format=self.data_format)
                x = batch_norm1d(x, is_training, data_format=self.data_format)
                x = tf.nn.relu(x)
        else:
            x = inputs
        with tf.variable_scope("pool"):
            if avg_or_max == "avg":
                x = tf.layers.average_pooling1d(
                    x, pool_size, strides, "SAME", data_format=actual_data_format)
            elif avg_or_max == "max":
                x = tf.layers.max_pooling1d(
                    x, pool_size, strides, "SAME", data_format=actual_data_format)
            else:
                raise ValueError("Unknown pool {}".format(avg_or_max))
        return lambda: x

    def _identity_branch(self, inputs):
        return lambda: inputs


class EnasConv1DwDataDescrption(EnasConv1dDAG):
    """This is a modeler that specifiied for convolution network with data description features
    Date: 2020.5.17
    """
    def __init__(self, data_description, *args, **kwargs):
        self.data_description = data_description
        super().__init__(*args, **kwargs)
        if len(self.data_description.shape) < 2:
            self.data_description = np.expand_dims(self.data_description, axis=0)

    # overwrite
    def _model(self, arc, **kwargs):
        """
        Overwrite the parent `_model` method to feed the description to controller when sampling architectures
        :param arc:
        :param kwargs:
        :return:
        """
        if self.train_fixed_arc:
            assert arc == self.fixed_arc or arc is None, "This DAG instance is built to train fixed arc, hence you " \
                                                         "can only provide arc=None or arc=self.fixed_arc; check the " \
                                                         "initialization of this instances "
        if arc is None:
            if self.train_fixed_arc:
                model = EnasCnnModel(inputs=self.fixed_model_input,
                                     outputs=self.fixed_model_output,
                                     labels=self.fixed_model_label,
                                     arc_seq=arc,
                                     dag=self,
                                     session=self.session,
                                     name=self.name)

            else:
                model = EnasCnnModel(inputs=self.sample_model_input,
                                     outputs=self.sample_model_output,
                                     labels=self.sample_model_label,
                                     arc_seq=arc,
                                     dag=self,
                                     session=self.session,
                                     name=self.name,
                                     sample_dag_feed_dict={
                                         self.controller.data_descriptive_feature: self.data_description}
                                     )
        else:
            model = EnasCnnModel(inputs=self.fixed_model_input,
                                 outputs=self.fixed_model_output,
                                 labels=self.fixed_model_label,
                                 arc_seq=arc,
                                 dag=self,
                                 session=self.session,
                                 name=self.name)
        return model
