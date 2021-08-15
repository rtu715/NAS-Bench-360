import warnings

import keras.backend as K
import numpy as np
import tensorflow as tf
if tf.__version__.startswith("2"):
    tf.compat.v1.disable_eager_execution()
    import tensorflow.compat.v1 as tf
from tensorflow.python.training import moving_averages


def unpack_data(data, unroll_generator_x=False, unroll_generator_y=False, callable_kwargs=None):
    is_generator = False
    unroll_generator = unroll_generator_x or unroll_generator_y
    if type(data) in (tuple, list):
        x, y = data[0], data[1]
    elif isinstance(data, tf.keras.utils.Sequence):
        x = data
        y = None
        is_generator = True
    elif hasattr(data, '__next__'):
        x = data
        y = None
        is_generator = True
    elif callable(data):
        callable_kwargs = callable_kwargs or {}
        x, y = unpack_data(data=data(**callable_kwargs),
                unroll_generator_x=unroll_generator_x,
                unroll_generator_y=unroll_generator_y)
    else:
        raise Exception("cannot unpack data of type: %s"%type(data))
    if is_generator and unroll_generator:
        gen = data if hasattr(data, '__next__') else iter(data)
        d_ = [d for d in zip(*gen)]
        if unroll_generator_x ^ unroll_generator_y:
            if hasattr(data, "shuffle"):
                assert data.shuffle == False
        x = np.concatenate(d_[0], axis=0) if unroll_generator_x else data
        y = np.concatenate(d_[1], axis=0) if unroll_generator_y else None
    return x, y


def batchify(x, y=None, batch_size=None, shuffle=True, drop_remainder=True):
    if not type(x) is list: x = [x]
    if y is not None and type(y) is not list: y = [y]
    # assuming batch_size is axis=0
    n = len(x[0])
    idx = np.arange(n)
    if batch_size is None:
        batch_size = n
    if shuffle:
        idx = np.random.choice(idx, n, replace=False)
    while True:
        for i in range(0, n, batch_size):
            tmp_x = [x_[idx[i:i + batch_size]] for x_ in x]
            if drop_remainder and tmp_x[0].shape[0] != batch_size:
                continue
            if y is not None:
                tmp_y = [y_[idx[i:i + batch_size]] for y_ in y]
                yield tmp_x, tmp_y
            else:
                yield tmp_x


def batchify_infer(x, y=None, batch_size=None, shuffle=True, drop_remainder=True):
    if not type(x) is list: x = [x]
    if y is not None and type(y) is not list: y = [y]
    # assuming batch_size is axis=0
    n = len(x[0])
    idx = np.arange(n)
    if batch_size is None:
        batch_size = n
    if shuffle:
        idx = np.random.choice(idx, n, replace=False)
    for i in range(0, n, batch_size):
        tmp_x = [x_[idx[i:i + batch_size]] for x_ in x]
        if drop_remainder and tmp_x[0].shape[0] != batch_size:
            continue
        if y is not None:
            tmp_y = [y_[idx[i:i + batch_size]] for y_ in y]
            yield tmp_x, tmp_y
        else:
            yield tmp_x

def numpy_shuffle_in_unison(List):
    rng_state = np.random.get_state()
    for x in List:
        np.random.set_state(rng_state)
        np.random.shuffle(x)


def get_tf_loss(loss, y_true, y_pred):
    loss = loss.lower()
    if loss == 'mse' or loss == 'mean_squared_error':
        loss_ = tf.reduce_mean(tf.square(y_true - y_pred))
    elif loss == 'categorical_crossentropy':
        loss_ = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
    elif loss == 'binary_crossentropy':
        loss_ = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    else:
        raise Exception("cannot understand string loss: %s" % loss)
    return loss_


def get_tf_metrics(m):
    if callable(m):
        return m
    elif m.lower() == 'mae':
        return tf.keras.metrics.MAE
    elif m.lower() == 'mse':
        return tf.keras.metrics.MSE
    elif m.lower() == 'acc':
        def acc(y_true, y_pred):
            return tf.reduce_mean(y_true)

        # return tf.keras.metrics.Accuracy
        return acc
    elif m.lower() == 'auc':
        return tf.keras.metrics.AUC
    else:
        raise Exception("cannot understand metric type: %s" % m)


def get_tf_layer(fn_str):
    fn_str = fn_str.lower()
    if fn_str == "relu":
        return tf.nn.relu
    elif fn_str == "linear":
        return lambda x: x
    elif fn_str == "softmax":
        return tf.nn.softmax
    elif fn_str == "sigmoid":
        return tf.nn.sigmoid
    elif fn_str == 'leaky_relu':
        return tf.nn.leaky_relu
    elif fn_str == 'elu':
        return tf.nn.elu
    elif fn_str == 'tanh':
        return tf.nn.tanh
    else:
        raise Exception("cannot get tensorflow layer for: %s" % fn_str)


def create_weight(name, shape, initializer=None, trainable=True, seed=None):
    if initializer is None:
        try:
            initializer = tf.contrib.keras.initializers.he_normal(seed=seed)
        except AttributeError:
            initializer = tf.keras.initializers.he_normal(seed=seed)
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)


def create_bias(name, shape, initializer=None):
    if initializer is None:
        initializer = tf.constant_initializer(0.0, dtype=tf.float32)
    return tf.get_variable(name, shape, initializer=initializer)


def batch_norm1d(x, is_training, name="bn", decay=0.9, epsilon=1e-5,
                 data_format="NWC"):
    if data_format == "NWC":
        shape = [x.get_shape()[-1]]
        x = tf.expand_dims(x, axis=1)  # NHWC
        sq_dim = 1
    elif data_format == "NCW":
        shape = [x.get_shape()[1]]
        x = tf.expand_dims(x, axis=2)  # NCHW
        sq_dim = 2
    else:
        raise NotImplementedError("Unknown data_format {}".format(data_format))

    with tf.variable_scope(name, reuse=False if is_training else True):
        offset = tf.get_variable(
            "offset", shape,
            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        scale = tf.get_variable(
            "scale", shape,
            initializer=tf.constant_initializer(1.0, dtype=tf.float32))
        moving_mean = tf.get_variable(
            "moving_mean", shape, trainable=False,
            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        moving_variance = tf.get_variable(
            "moving_variance", shape, trainable=False,
            initializer=tf.constant_initializer(1.0, dtype=tf.float32))

        if is_training:
            x, mean, variance = tf.nn.fused_batch_norm(
                x, scale, offset, epsilon=epsilon,
                is_training=True)
            update_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay)
            update_variance = moving_averages.assign_moving_average(
                moving_variance, variance, decay)
            with tf.control_dependencies([update_mean, update_variance]):
                x = tf.identity(x)
        else:
            x, _, _ = tf.nn.fused_batch_norm(
                x, scale, offset, mean=moving_mean,
                variance=moving_variance,
                epsilon=epsilon,
                is_training=False)
        x = tf.squeeze(x, axis=sq_dim)
    return x


def get_keras_train_ops(loss, tf_variables, optim_algo, **kwargs):
    assert K.backend() == 'tensorflow'
    # TODO: change to TF.keras
    from keras.optimizers import get as get_opt
    opt = get_opt(optim_algo)
    grads = tf.gradients(loss, tf_variables)
    grad_var = []
    no_grad_var = []
    for g, v in zip(grads, tf_variables):
        if g is None:
            # get sub-scope name; if is optimizer-related, ignore
            if 'compile' in v.name.split('/'):
                continue
            no_grad_var.append(v)
        else:
            grad_var.append(v)
    if no_grad_var:
        warnings.warn(
            "\n" + "=" * 80 + "\nWarning: the following tf.variables have no gradients"
                       " and have been discarded: \n %s" % no_grad_var, stacklevel=2)
    train_op = opt.get_updates(loss, grad_var)
    try:
        config = opt.get_config()
    except NotImplementedError:  # if cannot get learning-rate when eager-execution is disableed
        config = {'lr':None}
    try:
        learning_rate = config['lr']
    except:  # for newer version of keras
        learning_rate = config['learning_rate']
    return train_op, learning_rate, None, opt


def count_model_params(tf_variables):
    num_vars = 0
    for var in tf_variables:
        num_vars += np.prod([dim.value for dim in var.get_shape()])
    return num_vars


def proximal_policy_optimization_loss(curr_prediction, curr_onehot, old_prediction, old_onehotpred, rewards, advantage, clip_val, beta=None):
    rewards_ = tf.squeeze(rewards, axis=1)
    advantage_ = tf.squeeze(advantage, axis=1)

    entropy = 0
    r = 1
    for t, (p, onehot, old_p, old_onehot) in \
            enumerate(zip(curr_prediction, curr_onehot, old_prediction, old_onehotpred)):
        # print(t)
        # print("p", p)
        # print("old_p", old_p)
        # print("old_onehot", old_onehot)
        ll_t = tf.log(tf.reduce_sum(old_onehot * p))
        ll_0 = tf.log(tf.reduce_sum(old_onehot * old_p))
        r_t = tf.exp(ll_t - ll_0)
        r = r * r_t
        # approx entropy
        entropy += -tf.reduce_mean(tf.log(tf.reduce_sum(onehot * p, axis=1)))

    surr_obj = tf.reduce_mean(tf.abs(1 / (rewards_ + 1e-8)) *
                              tf.minimum(r * advantage_,
                                         tf.clip_by_value(r,
                                                          clip_value_min=1 - clip_val,
                                                          clip_value_max=1 + clip_val) * advantage_)
                              )
    if beta:
        # maximize surr_obj for learning and entropy for regularization
        return - surr_obj + beta * (- entropy)
    else:
        return - surr_obj


def get_kl_divergence_n_entropy(curr_prediction, curr_onehot, old_prediction, old_onehotpred):
    """compute approx
    return kl, ent
    """
    kl = []
    ent = []
    for t, (p, onehot, old_p, old_onehot) in \
            enumerate(zip(curr_prediction, curr_onehot, old_prediction, old_onehotpred)):
        # print(t, old_p, old_onehot, p, onehot)
        kl.append(tf.reshape(tf.keras.metrics.kullback_leibler_divergence(old_p, p), [-1]))
        ent.append(tf.reshape(tf.keras.backend.binary_crossentropy(onehot, p), [-1]))
    return tf.reduce_mean(tf.concat(kl, axis=0)), tf.reduce_mean(tf.concat(ent, axis=0))


def lstm(x, prev_c, prev_h, w):
    ifog = tf.matmul(tf.concat([x, prev_h], axis=1), w)
    i, f, o, g = tf.split(ifog, 4, axis=1)
    i = tf.sigmoid(i)
    f = tf.sigmoid(f)
    o = tf.sigmoid(o)
    g = tf.tanh(g)
    next_c = i * g + f * prev_c
    next_h = o * tf.tanh(next_c)
    return next_c, next_h


def stack_lstm(x, prev_c, prev_h, w):
    next_c, next_h = [], []
    for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
        inputs = x if layer_id == 0 else next_h[-1]
        curr_c, curr_h = lstm(inputs, _c, _h, _w)
        next_c.append(curr_c)
        next_h.append(curr_h)
    return next_c, next_h
