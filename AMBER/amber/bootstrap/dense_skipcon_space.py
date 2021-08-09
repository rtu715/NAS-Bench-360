# -*- coding: UTF-8 -*-
"""
This module's documentation is Work-In-Progress.

This module might be deprecated in the future.
"""


import numpy as np

from ..architect import GeneralManager
from ..modeler import DAGModelBuilder
# for model space
from ..architect import State, ModelSpace
from ..architect.store import store_with_hessian
# for controller
from ..architect.reward import KnowledgeReward
# for Hess
from ..objective import GraphKnowledgeHessFunc
from ..utils.simulator import HigherOrderSimulator, CorrelatedDataSimulator


def get_model_space(num_layers):
    state_space = ModelSpace()
    for i in range(num_layers):
        state_space.add_layer(i, [
            State('Dense', units=5, activation='relu'),
            State('Dense', units=5, activation='tanh')
        ])
    return state_space


def get_input_nodes(num_inputs, with_input_blocks):
    input_state = []
    if with_input_blocks:
        for node in range(num_inputs):
            units = 1
            name = 'X%i' % node
            # input_state.append(get_layer(None, State('input', shape=(units,), name=name)))
            node_op = State('input', shape=(units,), name=name)
            input_state.append(node_op)
    else:
        input_state = [State('input', shape=(num_inputs,), name='Input')]
    return input_state


def get_output_nodes():
    output_op = State('Dense', units=1, activation='linear', name='output')
    # output_node = ComputationNode(output_op, node_name='output')
    return output_op


def get_data(with_input_blocks):
    # set global random seed
    np.random.seed(111)
    n = 5000
    p = 4
    # Y = f(X0,X1,X2,X3 = 3*X0X1 - 2*X2X3
    beta_a = np.array([0, 0, 0, 0]).astype('float32')
    beta_i = np.array([0, 3, 0, 0] + [0] * 3 + [0, -2] + [0]).astype('float32')

    simulator = HigherOrderSimulator(n=n, p=p,
                                     noise_var=0.1,
                                     x_var=1.,
                                     degree=2,
                                     discretize_beta=True,
                                     discretize_x=False,
                                     with_input_blocks=with_input_blocks)
    simulator.set_effect(beta_a, beta_i)
    X_train, y_train = simulator.sample_data()
    X_val, y_val = simulator.sample_data(N=500)
    X_test, y_test = simulator.sample_data(N=500)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def get_data_correlated(with_input_blocks, corr_coef=0.6):
    # set global random seed
    np.random.seed(111)
    n = 5000
    p = 4
    # Y = f(X0,X1,X2,X3 = 3*X0X1 - 2*X2X3
    # cor(X0, X2) = 0.6
    beta_a = np.array([0, 0, 0, 0]).astype('float32')
    beta_i = np.array([0, 3, 0, 0] + [0] * 3 + [0, -2] + [0]).astype('float32')

    cov_mat = np.eye(4) * 1.
    cov_mat[0, 2] = cov_mat[2, 0] = 1. * corr_coef
    simulator = CorrelatedDataSimulator(n=n, p=p,
                                        noise_var=0.1,
                                        data_cov_matrix=cov_mat,
                                        degree=2,
                                        discretize_beta=True,
                                        with_input_blocks=with_input_blocks)
    simulator.set_effect(beta_a, beta_i)
    X_train, y_train = simulator.sample_data()
    X_val, y_val = simulator.sample_data(N=500)
    X_test, y_test = simulator.sample_data(N=500)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def get_knowledge_fn():
    gkf = GraphKnowledgeHessFunc(total_feature_num=4)
    adjacency = np.zeros((4, 4))
    adjacency[0, 1] = adjacency[1, 0] = 3.
    adjacency[2, 3] = adjacency[3, 2] = -2.
    intr_idx, intr_eff = gkf.convert_adjacency_to_knowledge(adjacency)
    gkf.knowledge_encoder(intr_idx, intr_eff)
    return gkf


def get_reward_fn(gkf, Lambda=1.):
    reward_fn = KnowledgeReward(gkf, Lambda=Lambda)
    return reward_fn


def get_manager(train_data, validation_data, model_fn, reward_fn, wd='./tmp'):
    model_compile_dict = {'loss': 'mse', 'optimizer': 'adam', 'metrics': ['mae']}
    manager = GeneralManager(train_data, validation_data,
                             working_dir=wd,
                             model_fn=model_fn,
                             reward_fn=reward_fn,
                             post_processing_fn=store_with_hessian,
                             model_compile_dict=model_compile_dict,
                             epochs=100, verbose=0,
                             child_batchsize=100
                             )
    return manager


def get_model_fn(model_space,
                 inputs_op,
                 output_op,
                 num_layers,
                 with_skip_connection,
                 with_input_blocks):
    model_compile_dict = {'loss': 'mse', 'optimizer': 'adam', 'metrics': ['mae']}
    model_fn = DAGModelBuilder(inputs_op, output_op, num_layers, model_space, model_compile_dict,
                               with_skip_connection, with_input_blocks,
                               dag_func='InputBlockDAG')
    return model_fn
