# -*- coding: UTF-8 -*-

"""top-k elements and masking the rest to zero"""

import tensorflow as tf


def sparsek_vec(x):
    convmap = x
    shape = tf.shape(convmap)
    nb = shape[0]
    nl = shape[1]
    tk = tf.cast(0.2 * tf.cast(nl, tf.float32), tf.int32)

    convmapr = tf.reshape(convmap, tf.stack([nb, -1]))  # convert input to [batch, -1]

    th, _ = tf.nn.top_k(convmapr, tk)  # nb*k
    th1 = tf.slice(th, [0, tk - 1], [-1, 1])  # nb*1 get kth threshold
    thr = tf.reshape(th1, tf.stack([nb, 1]))
    drop = tf.where(convmap < thr,
                    tf.zeros([nb, nl], tf.float32), tf.ones([nb, nl], tf.float32))
    convmap = convmap * drop

    return convmap

