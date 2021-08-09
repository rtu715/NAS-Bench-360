# -*- coding: UTF-8 -*-

import itertools
import warnings

import keras.backend as K
import numpy as np
import tensorflow as tf

from .generalObjFunc import GeneralKnowledgeObjectiveFunction
from .generalObjMath import bias_var_decomp


class GraphHierarchyTree(GeneralKnowledgeObjectiveFunction):
    """GraphHierarchyTree measures the clustering cost function for a given
    undirected graph. The hierarchy must be generated with ```unique_input_connection=True```.


    Parameters
    -----------
    input_blocks : list of str
        a list of the input node names

    total_feature_num : int
        number of total features

    block_index_mapping : dict
        maps input block names (str) to the index (int) of the flattened feature list
        Flattened feature list = tf.concat(model.inputs, axis=0)

    Examples
    --------
    An exemplar call for GHT knowledge function.
    .. code-block::

        # import functions from MockBlackBox
        from BioNAS.MockBlackBox.dense_skipcon_space import get_data, get_data_correlated, get_model_space, \
            get_manager, get_reward_fn, get_knowledge_fn, get_model_fn, get_input_nodes, get_output_nodes
        inputs_op = get_input_nodes(4, with_input_blocks)
        output_op = get_output_nodes()

        model_fn = get_model_fn(model_space, inputs_op, output_op, num_layers,
                                with_skip_connection=with_skip_connection,
                                with_input_blocks=with_input_blocks)

        # architecture for num_layers=3, num_input_blocks=4
        arc = np.array([1, 1,1,0,0,
                        1, 0,0,1,1, 0,
                        1, 0,0,0,0, 1,1])
        model = model_fn(arc)

        ght = GraphHierarchyTree(
            input_blocks=['X%i'%i for i in range(4)],
            total_feature_num=4,
            block_index_mapping={'X%i'%i:[i] for i in range(4)}
        )

        # distance = 1 - adjacency
        adjacency = np.zeros((4,4))
        adjacency[0,1] = adjacency[1,0] = 1
        adjacency[2,3] = adjacency[3,2] = 1
        ght.knowledge_encoder(adjacency)

        # call knowledge function to get dissimilarity score
        k = ght(model, None)
        print(k) # k = 0.5
        print(ght.W_model) # W_model is the pair-wise leave cardinality


    """

    def __init__(self, input_blocks, total_feature_num, block_index_mapping, *args, **kwargs):
        self.input_blocks = input_blocks
        self.total_feature_num = total_feature_num
        self.block_index_mapping = block_index_mapping

        self.nodes = None
        self.W_model = None
        self.W_knowledge = {}

        self._build_obj_func()

    def __call__(self, model, data, **kwargs):
        return super(GraphHierarchyTree, self).__call__(model, data, **kwargs)

    def __str__(self):
        s = 'GraphHierarchyTree'
        return s

    def _get_model_config(self, model):
        # self.input_blocks = [inp.name.split(':')[0] for inp in model.inputs]
        self.nodes = model.nodes

    def _get_tree_roots(self):
        # inp_head keeps a queue of all leading branches for each input block
        inp_heads = {inp: None for inp in self.input_blocks}
        # inp_pair_root keeps the tree subroot for each pair of input blocks
        inp_pair_roots = {(b1, b2): None for b1 in self.input_blocks for b2 in self.input_blocks}
        # root_leaves keeps all leaves for all nodes
        leaves_cardinality = {n.node_name: set([]) for n in self.nodes}
        for n in self.nodes:
            # get inputs to this layer and update heads
            _inputs = [x.node_name for x in n.parent if x.operation.Layer_type == 'input']
            inp_heads.update({x: n.node_name for x in _inputs})

            # get the set of parents nodes that are not input_blocks
            _ops = set([x.node_name for x in n.parent if x.operation.Layer_type != 'input'])

            # update leave cardinality
            for leaf in _inputs + [l for x in _ops for l in leaves_cardinality[x]]:
                leaves_cardinality[n.node_name].add(leaf)

            # update heads if connected to this layer
            inp_heads.update({x: n.node_name for x in self.input_blocks if inp_heads[x] in _ops})

            # update inp_pair_roots if new inp_heads met each other
            for b1 in self.input_blocks:
                for b2 in self.input_blocks:
                    if inp_pair_roots[(b1, b2)] is not None:
                        continue
                    head1 = inp_heads[b1]
                    head2 = inp_heads[b2]
                    if head1 == head2 == n.node_name:
                        inp_pair_roots[(b1, b2)] = n.node_name

        return inp_heads, inp_pair_roots, leaves_cardinality

    def model_encoder(self, model, data, *args, **kwargs):
        self._get_model_config(model)
        _, inp_pair_roots, leaves_cardinality = self._get_tree_roots()
        self.W_model = {}
        for idx in inp_pair_roots:
            if inp_pair_roots[idx]:
                self.W_model[idx] = len(leaves_cardinality[inp_pair_roots[idx]])
            else:
                self.W_model[idx] = len(model.inputs)

    def knowledge_encoder(self, adjacency):
        """
        Parameters
        ----------
        adjacency:

        Returns
        -------
        None

        Notes
        -----
            adjacency is on the feature-level, not block-level; in the context of
            biology, adjacency is for genes, not for GO terms/blocks

        """
        assert adjacency.shape[0] == adjacency.shape[
            1] == self.total_feature_num, "adjacency shape not match total_feature_num"
        if np.min(adjacency) < 0:
            warnings.warn("detected negative adjacency for GraphHierarchyTree; \
                          converting to absolute values..")
            adjacency = np.abs(adjacency)
        self.W_knowledge = np.copy(adjacency)
        # for i in range(self.total_feature_num):
        #    for j in range(i, self.total_feature_num):
        #        if j == i:
        #            continue
        #        self.W_knowledge[(i,j)] = adjacency[i, j]
        return

    def _build_obj_func(self, **kwargs):
        def obj_fn(W_model, W_knowledge, *args, **kwargs):
            k = np.zeros((self.total_feature_num, self.total_feature_num))
            for b1 in self.input_blocks:
                for b2 in self.input_blocks:
                    if b1 == b2: continue
                    leave_cardinality = W_model[(b1, b2)]
                    f1 = self.block_index_mapping[b1]
                    f2 = self.block_index_mapping[b2]
                    for idx in itertools.product(f1, f2):
                        k[idx] = W_knowledge[idx] * leave_cardinality
            k_sum = np.mean(k)
            return k_sum

        self.obj_fn = obj_fn
        return


class GraphHierarchyTreeAuxLoss(GraphHierarchyTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_loss_mapping = None

    def __str__(self):
        return 'GraphHierarchyTree with AuxLoss'

    def _get_model_config(self, model):
        assert model.block_loss_mapping is not None, "AuxLoss must have model.block_loss_mapping; did you use " \
                                                     "InputBlockAuxLossDAG` in model_builder?"
        self.nodes = model.nodes
        self.block_loss_mapping = model.block_loss_mapping

    def model_encoder(self, model, data, *args, **kwargs):
        self._get_model_config(model)
        _, inp_pair_roots, leaves_cardinality = self._get_tree_roots()
        all_losses = model.evaluate(data[0], data[1], final_only=False)
        epsilon = 0.0001

        def linobj_func(x):
            if x is None:
                return 1.
            g = 1. if -epsilon < (x - all_losses[1]) < epsilon else np.tanh(1. / (x - all_losses[1]))
            return g

        self.W_model = {}
        for idx in inp_pair_roots:
            if inp_pair_roots[idx]:
                self.W_model[idx] = len(leaves_cardinality[inp_pair_roots[idx]]) * linobj_func(
                    self.block_loss_mapping[idx])
            else:
                self.W_model[idx] = len(model.inputs)

    def _build_obj_func(self, **kwargs):
        def obj_fn(W_model, W_knowledge, *args, **kwargs):
            k = np.zeros((self.total_feature_num, self.total_feature_num))
            for b1 in self.input_blocks:
                for b2 in self.input_blocks:
                    if b1 == b2: continue
                    leave_cardinality = W_model[(b1, b2)]
                    f1 = self.block_index_mapping[b1]
                    f2 = self.block_index_mapping[b2]
                    for idx in itertools.product(f1, f2):
                        k[idx] = W_knowledge[idx] * leave_cardinality
            k_sum = np.mean(k)
            return k_sum

        self.obj_fn = obj_fn
        return


class GraphKnowledgeHessFunc(GeneralKnowledgeObjectiveFunction):
    def __init__(self, total_feature_num=None):
        # for model_encoder
        self.session = None
        self.total_feature_num = total_feature_num
        self.is_hess_built = False
        self.hess_op = None
        self.jacob_op = None
        # for knowledge_encoder
        # for calling
        self.W_knowledge = {}
        self.W_model = None
        self._build_obj_func()

    def __call__(self, model, data, **kwargs):
        return super(GraphKnowledgeHessFunc, self).__call__(model, data, **kwargs)

    def __str__(self):
        return 'Graph K-function for Interpretable Model Learning'

    def model_encoder(self, model, data, **kwargs):
        self.is_hess_built = False  # need to rebuild Hess
        try:
            g, h = self.hess_estimator(model, data, **kwargs)
            self.W_model = h
        except:  # TODO: figure out more specific Exceptions
            # self.W_model = np.ones( (10, self.total_feature_num, self.total_feature_num) )
            self.W_model = np.random.normal(0, 1, 10 * self.total_feature_num ** 2).reshape(
                (10, self.total_feature_num, self.total_feature_num))
        return

    def hess_estimator(self, model, data, **kwargs):
        # have to get session before run
        if (not self.is_hess_built) or (not self.session):
            try:
                self.session = kwargs['session']
            except KeyError:
                print("try getting session from Keras..")
                self.session = K.get_session()
            finally:
                assert 'session' in kwargs, 'GraphKnowledge requires passing the Keras session;' \
                                            'check here for details: https://github.com/tensorflow/tensorflow/issues/26472'
        # build hess_op input/output tensors
        if not self.is_hess_built:
            xs = [model.inputs[i] for i in range(len(model.inputs))]
            ys = [model.outputs[i] for i in range(len(model.outputs)) if
                  not model.outputs[i].name.startswith('added_out')]
            if self.total_feature_num is None:
                self.total_feature_num = sum([t.shape[1].value for t in model.inputs])
            self.jacob_op, self.hess_op = self.get_hess_op(ys, xs)
            self.is_hess_built = True
        # build feed_dict
        feed_dict = self.get_feed_dict(model, data)
        # run graph
        jacob, hess = self.session.run([self.jacob_op, self.hess_op], feed_dict)
        return jacob, hess

    def get_feed_dict(self, model, data):
        # provide feed_dict
        feed_dict = {}
        xs, ys = data
        if len(model.inputs) > 1:
            feed_dict.update({model.inputs[i]: xs[i] for i in range(len(model.inputs))})
        else:
            feed_dict.update({model.inputs[0]: xs})

        if len(model.outputs) > 1:
            data_idx = 0
            if type(ys) is not list:
                ys = [ys]
            for i in range(len(model.outputs)):
                if not model.outputs[i].name.startswith('added_out'):
                    # print(data_idx, i, model.outputs[i])
                    feed_dict.update({
                        model.outputs[i]: ys[data_idx].reshape((-1, model.outputs[i].shape[1].value))
                    })
                    data_idx += 1
        else:
            feed_dict.update({model.outputs[0]: ys.reshape((-1, model.outputs[0].shape[1]))})
        return feed_dict

    def get_hess_op(self, ys, xs):
        _gradients = tf.concat(tf.gradients(ys, xs), axis=1)
        _xs = tf.concat(xs, axis=1)

        hessians = []
        for j in range(self.total_feature_num):
            if not j % 10:
                print("building hess %i / %i" % (j, self.total_feature_num))
            hessians.append(tf.concat(
                tf.gradients(_gradients[:, j],
                             xs), axis=1))
        hess_op = tf.transpose(tf.stack(hessians), [1, 0, 2])
        return _gradients, hess_op

    def knowledge_encoder(self, intr_idx, intr_eff, **kwargs):
        """encoding a partial graph/adjacency matrix from prior knowledge

        Parameters
        ----------
        intr_idx: list
             a List of length-of-2 tuples, each tuple is
            the indices for two X features
        intr_eff: list
            a List of interaction effect size; inverse of distance

        Returns
        -------
            None
        """
        for idx, eff in zip(intr_idx, intr_eff):
            assert len(idx) == 2, "interaction idx ``intr_idx`` must be size-2 tuples; got %s" % idx
            self.W_knowledge[tuple(idx)] = eff
        return

    def convert_adjacency_to_knowledge(self, adjacency):
        assert adjacency.shape[0] == adjacency.shape[1] == self.total_feature_num
        intr_idx = []
        intr_eff = []
        for i in range(self.total_feature_num):
            for j in range(i, self.total_feature_num):
                if j == i:
                    continue
                intr_idx.append((i, j))
                intr_eff.append(adjacency[i, j])
        return intr_idx, intr_eff

    def _build_obj_func(self, **kwargs):
        def obj_fn(W_model, W_knowledge, *args, **kwargs):
            score = 0
            i = 0
            for idx, eff in W_knowledge.items():
                i += 1
                # use bias^2 + var
                score += bias_var_decomp(eff, W_model[:, idx[0], idx[1]])
                # use just bias for benchmarking.
                # score += (eff - np.mean(W_model[:, idx[0], idx[1]]))**2
            return score / i

        self.obj_fn = obj_fn
        return

    def _reset(self):
        del self.jacob_op, self.hess_op
        self.jacob_op = None
        self.hess_op = None
        if self.session is not None:
            # self.session.close()
            # del self.session
            self.session = None


class GraphKnowledgeHessBias(GraphKnowledgeHessFunc):
    def _build_obj_func(self, **kwargs):
        def obj_fn(W_model, W_knowledge, *args, **kwargs):
            score = 0
            i = 0
            for idx, eff in W_knowledge.items():
                i += 1
                # use just bias for benchmarking.
                score += (eff - np.mean(W_model[:, idx[0], idx[1]])) ** 2
            return score / i

        self.obj_fn = obj_fn
        return
