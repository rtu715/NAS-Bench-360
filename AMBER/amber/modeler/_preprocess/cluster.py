# -*- coding: utf-8 -*-

"""
Performs feature grouping into input/output blocks for
a variety of base model architectures
ZZJ, 11.25.2019
"""

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans


def build_feature_model_from_cnn(model, target_layer=None):
    layer_dict = {l.name: l for l in model.layers}

    if target_layer is None:
        target_layer = sorted([k for k in layer_dict if k.startswith('conv')])[-1]
    model2 = Model(inputs=model.inputs, outputs=layer_dict[target_layer].output)
    model2.compile(optimizer='adam', loss='mse')
    return model2


def cluster_cnn_channels(model, x, num_input_blocks, out_fn, target_layer=None):
    model2 = build_feature_model_from_cnn(model, target_layer)
    pred = model2.predict(x)

    n_channels = model2.output.shape[-1].value
    # transpose to shape: channel, sample, spatial feature
    flatten_feature_map = np.transpose(pred, [2, 0, 1]).reshape([n_channels, -1])
    # np.save('feature_map.npy', flatten_feature_map)

    # NOTE: sklearn KMeans already run 20 times and selects the best clustering results
    kmeans = KMeans(n_clusters=num_input_blocks,
                    n_jobs=-1,
                    random_state=1710,
                    n_init=20)
    f_kmean_assign = kmeans.fit_predict(flatten_feature_map)
    np.save(out_fn, f_kmean_assign)
    return f_kmean_assign


def split_multitask_labels(y, split_vec):
    ys = []
    for v in split_vec:
        ys.append(y[:, v])
    return ys


def make_biosequence_tfrecords(x, y, filename):
    assert type(x) is list
    assert type(y) is list
    num_samples = x[0].shape[0]
    num_bp = x[0].shape[1]
    num_in = len(x)
    num_out = len(y)
    output_block_dims = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[y[i].shape[1]]))
        for i in range(num_out)
    ]
    x_feature = [tf.train.Feature(bytes_list=tf.train.BytesList(value=x[i].reshape(-1).astype('bytes')))
                 for i in range(num_in)]
    y_feature = [tf.train.Feature(bytes_list=tf.train.BytesList(value=y[i].reshape(-1).astype('bytes')))
                 for i in range(num_out)]
    feature_key_value_pair = {
        'num_samples': tf.train.Feature(int64_list=tf.train.Int64List(value=[num_samples])),
        'num_bp': tf.train.Feature(int64_list=tf.train.Int64List(value=[num_bp])),
        'num_out': tf.train.Feature(int64_list=tf.train.Int64List(value=[num_out])),
        'num_in': tf.train.Feature(int64_list=tf.train.Int64List(value=[num_in])),
    }
    feature_key_value_pair.update({
        'x_%i' % i: x_feature[i] for i in range(num_in)
    })
    feature_key_value_pair.update({
        'y_%i' % i: y_feature[i] for i in range(num_out)
    })
    feature_key_value_pair.update({
        'out_dim_%i' % i: output_block_dims[i] for i in range(num_out)
    })
    features = tf.train.Features(feature=feature_key_value_pair)
    example = tf.train.Example(features=features)
    with tf.python_io.TFRecordWriter(filename) as tfwritter:
        tfwritter.write(example.SerializeToString())


def read_pred(fn):
    # fn = "weights/trial_0/pred.txt"
    import pandas as pd
    import sklearn.metrics as metrics
    import numpy as np
    df = pd.read_csv(fn, sep="\t")
    pred = df['pred'].str.split(',', expand=True).to_numpy().astype('float')
    obs = df['obs'].str.split(',', expand=True).to_numpy().astype('int')
    auc = [metrics.roc_auc_score(y_true=obs[:, i], y_score=pred[:, i])
           if sum(obs[:, i]) > 0 else np.nan
           for i in range(obs.shape[1])]

    return auc
