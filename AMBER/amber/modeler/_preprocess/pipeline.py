# -*- coding: utf-8 -*-

"""
Pipelining the grouped input features into
ZZJ, 11.25.2019
"""

import numpy as np
import tensorflow as tf

from ...architect import State


def get_features_in_block(spatial_outputs, f_kmean_assign):
    n_clusters = np.max(f_kmean_assign) + 1

    block_ops = []
    for c in range(n_clusters):
        i = np.where(f_kmean_assign == c)[0]
        block_ops.append(tf.gather(spatial_outputs, indices=i, axis=-1, batch_dims=0))
    return block_ops


class CnnFeatureModel:
    """
    Convert the last Conv layer in a Cnn model as features to a Dnn model
    Keep all Tensors for future Operations
    """

    def __init__(self, base_model, session, feature_assign_fn,
                 feature_map_orientation='channels_last',
                 x=None, y=None, batch_size=128, target_layer=None,
                 trainable=None,
                 name='CnnFeatureModel'):
        self.name = name
        layer_dict = {l.name: l for l in base_model.layers}
        base_model.trainable = trainable or False

        if target_layer is None:
            target_layer = sorted([k for k in layer_dict if k.startswith('conv')])[-1]

        self.session = session
        self.base_model = base_model
        # x_inputs: the actual inputs from data
        self.x_inputs = base_model.inputs
        self.spatial_outputs = layer_dict[target_layer].output
        self.total_feature_num = np.prod(self.spatial_outputs.shape[1:]).value
        # because we will mask by channel, we will re-arrange such that features in
        # the same channel will be close together ZZ 2019.12.4
        if feature_map_orientation == 'channels_last':
            self.orient = [0, 2, 1]
        elif feature_map_orientation == 'channels_first':
            self.orient = [0, 1, 2]
        else:
            raise Exception("cannot understand feature_map_orientation: %s" % feature_map_orientation)
        self.outputs = tf.reshape(tf.transpose(self.spatial_outputs, self.orient), [-1, self.total_feature_num])
        self.load_feature_blocks(feature_assign_fn)
        self.batch_size = None
        self.data_gen = None
        # _it are Tensor Iterators
        self.x_it = None
        self.y_it = None
        # _ph are Numpy Array for all data
        self.x_ph = None
        self.y_ph = None
        # tensorflow dataset pseudo_input
        self.pseudo_inputs_pipe = None

    # re-initializable & feedable iterator
    def get_data_pipeline_feedable(self, label_shapes, batch_size=None):
        if batch_size is None:
            batch_size = 32
        self.x_ph = [
            tf.placeholder(shape=self.base_model.inputs[i].shape, dtype=np.float32, name="base_input_%i" % i)
            for i in range(len(self.base_model.inputs))]
        self.y_ph = [
            tf.placeholder(shape=label_shapes[i], dtype=np.float32, name="base_output_%i" % i)
            for i in range(len(label_shapes))]
        self.batch_size = batch_size
        dataset = tf.data.Dataset.from_tensor_slices(tuple(self.x_ph + self.y_ph))
        self.data_gen = dataset.repeat().shuffle(self.batch_size).batch(self.batch_size).make_initializable_iterator()
        # do not shuffle if for debug
        # self.data_gen = dataset.batch(self.batch_size).make_initializable_iterator()
        next_ele = self.data_gen.get_next()
        self.x_it = list(next_ele[:len(self.x_ph)])
        self.y_it = list(next_ele[len(self.x_ph)::])
        self.base_model.layers.pop(0)
        # spatial_outputs_pipe = tf.nn.dropout(self.base_model(self.x_it),
        #                                     keep_prob=1-self.dropout_rate)
        spatial_outputs_pipe = self.base_model(self.x_it)
        block_ops = get_features_in_block(spatial_outputs_pipe, self.f_assign)
        # pseudo_inputs: after tensor processing of x_input, the output is fed as "pseudo"-input into dense nn
        self.pseudo_inputs_pipe = tf.concat(
            [tf.reshape(tf.transpose(x, self.orient), [-1, np.prod(x.shape[1:]).value]) for x in block_ops], axis=1)

    def predict(self, x_, keep_spatial=True):
        if type(x_) is not list:
            x_ = [x_]
        if keep_spatial:
            return self.session.run(self.spatial_outputs, feed_dict={self.x_inputs[i]: x_[i] for i in range(len(x_))})
        else:
            return self.session.run(self.outputs, feed_dict={self.x_inputs[i]: x_[i] for i in range(len(x_))})

    def load_feature_blocks(self, feature_assign_fn):
        f_assign = np.load(feature_assign_fn)
        block_ops = get_features_in_block(self.spatial_outputs, f_assign)
        self.input_blocks = [tf.reshape(tf.transpose(x, self.orient), [-1, np.prod(x.shape[1:]).value]) for x in
                             block_ops]
        # pseudo_inputs: after tensor processing of x_input, the output is fed as "pseudo"-input into dense nn
        self.pseudo_inputs = tf.concat(self.input_blocks, axis=1)
        # input_node: instances provided for downstream NAS
        self.input_node_for_nas = [
            State('input', shape=(self.input_blocks[i].shape[1].value,), name='Input_%i' % i)
            for i in range(len(block_ops))]
        self.f_assign = f_assign
