import determined as det
from determined import experimental
from determined.experimental.keras import init

config = {
        "searcher": {"name": "single", "metric": "val_acc", "max_length": {'epochs': 2} },
    "hyperparameters": {"global_batch_size": 32},
    "records_per_epoch": 50000
}

import itertools
import os
import pickle
import sys
import copy
import numpy as np
import tensorflow as tf
from argparse import Namespace

os.environ['search_space'] = 'darts'
from data import Data
from params import *

context = init(config, test=False, local=False, context_dir=".")


metann_params = meta_neuralnet_params('darts')
algorithm_params = algo_params('bananas')
mp = copy.deepcopy(metann_params)
ss = mp.pop('search_space')

dataset = mp.pop('dataset')
#doesn't matter what dataset is in darts space, defined in darts repo
search_space = Data(ss, dataset=dataset)
ps = copy.deepcopy(algorithm_params)


from acquisition_functions import acq_fn
from meta_neural_net import MetaNeuralnet

num_init=10 
k=10
loss='val_loss'
total_queries=150 
num_ensemble=5
acq_opt_type='mutation'
num_arches_to_mutate=1
explore_type='its'
encoding_type='trunc_path'
cutoff=40
deterministic=True
verbose=1


data = search_space.generate_random_dataset(num=num_init, 
                                            encoding_type=encoding_type, 
                                            cutoff=cutoff,
                                            deterministic_loss=deterministic)

query = num_init + k

while query <= total_queries:

    xtrain = np.array([d['encoding'] for d in data])
    ytrain = np.array([d[loss] for d in data])


    # get a set of candidate architectures
    candidates = search_space.get_candidates(data, 
                                             acq_opt_type=acq_opt_type,
                                             encoding_type=encoding_type, 
                                             cutoff=cutoff,
                                             num_arches_to_mutate=num_arches_to_mutate,
                                             loss=loss,
                                             deterministic_loss=deterministic)

    xcandidates = np.array([c['encoding'] for c in candidates])
    candidate_predictions = []

    # train an ensemble of neural networks
    train_error = 0
    for _ in range(num_ensemble):
        meta_neuralnet = MetaNeuralnet().get_dense_model((xtrain.shape[1],),
                **metann_params)

        model = context.wrap_model(meta_neuralnet)
        optimizer = keras.optimizers.Adam(lr=lr, beta_1=.9, beta_2=.99)

        model.compile(optimizer=optimizer, loss='mape')

        train_error += model.fit(xtrain, ytrain, **metann_params)

        # predict the validation loss of the candidate architectures
        candidate_predictions.append(np.squeeze(meta_neuralnet.predict(xcandidates)))

        # clear the tensorflow graph
        tf.reset_default_graph()

    tf.keras.backend.clear_session()

    train_error /= num_ensemble

    # compute the acquisition function for all the candidate architectures
    candidate_indices = acq_fn(candidate_predictions, explore_type)

    # add the k arches with the minimum acquisition function values
    for i in candidate_indices[:k]:

        arch_dict = search_space.query_arch(candidates[i]['spec'],
                                            encoding_type=encoding_type,
                                            cutoff=cutoff,
                                            deterministic=deterministic)
        data.append(arch_dict)

    query += k


