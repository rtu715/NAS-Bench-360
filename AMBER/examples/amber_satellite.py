#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example wrapper `Amber` use for searching PhysioNet challenge data
"""

from amber import Amber
from amber.architect import ModelSpace, Operation
from load_satellite import load_satellite_data 

def get_model_space(out_filters=64, num_layers=9):
    model_space = ModelSpace()
    num_pool = 4
    expand_layers = [num_layers//4-1, num_layers//4*2-1, num_layers//4*3-1]
    for i in range(num_layers):
        model_space.add_layer(i, [
            Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu'),
            Operation('conv1d', filters=out_filters, kernel_size=4, activation='relu'),
            #Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu', dilation=10),
            #Operation('conv1d', filters=out_filters, kernel_size=4, activation='relu', dilation=10),
            # max/avg pool has underlying 1x1 conv
            Operation('maxpool1d', filters=out_filters, pool_size=4, strides=1),
            Operation('avgpool1d', filters=out_filters, pool_size=4, strides=1),
            Operation('identity', filters=out_filters),
      ])
        if i in expand_layers:
            out_filters *= 2
    return model_space


# First, define the components we need to use
type_dict = {
    'controller_type': 'GeneralController',
    'modeler_type': 'EnasCnnModelBuilder',
    'knowledge_fn_type': 'zero',
    'reward_fn_type': 'LossAucReward',
    'manager_type': 'EnasManager',
    'env_type': 'EnasTrainEnv'
}


# Next, define the specifics
wd = "./outputs/AmberECG/"
X_train, Y_train, X_val, Y_val = load_satellite_data('.', True)
train_data = (X_train, Y_train)
val_data = (X_val, Y_val)
input_node = Operation('input', shape=(46, 1), name="input")
output_node = Operation('dense', units=24, activation='sigmoid')
model_compile_dict = {
    'loss': 'categorical_crossentropy',
    'optimizer': 'adam',
}
model_space = get_model_space(out_filters=32, num_layers=12)

specs = {
    'model_space': model_space,
    
    'controller': {
            'share_embedding': {i:0 for i in range(1, len(model_space))},
            'with_skip_connection': True,
            'num_input_blocks': 1,
            'skip_connection_unique_connection': False,
            'skip_weight': 1.0,
            'skip_target': 0.4,
            'lstm_size': 64,
            'lstm_num_layers': 1,
            'kl_threshold': 0.01,
            'train_pi_iter': 10,
            'optim_algo': 'adam',
            'temperature': 2.,
            'lr_init': 0.001,
            'tanh_constant': 1.5,
            'buffer_size': 1,  
            'batch_size': 20
    },

    'model_builder': {
        'dag_func': 'EnasConv1dDAG',
        'batch_size': 1000,
        'inputs_op': [input_node],
        'outputs_op': [output_node],
        'model_compile_dict': model_compile_dict,
         'dag_kwargs': {
            'stem_config': {
                'flatten_op': 'flatten',
                'fc_units': 925
            }
        }
    },

    'knowledge_fn': {'data': None, 'params': {}},

    'reward_fn': {'method': 'auc'},

    'manager': {
        'data': {
            'train_data': train_data,
            'validation_data': val_data
        },
        'params': {
            'epochs': 1,
            'child_batchsize': 1000,
            'store_fn': 'minimal',
            'working_dir': wd,
            'verbose': 2
        }
    },

    'train_env': {
        'max_episode': 300,
        'max_step_per_ep': 100,
        'working_dir': wd,
        'time_budget': "24:00:00",
        'with_input_blocks': False,
        'with_skip_connection': True,
        'child_train_steps': 500,
        'child_warm_up_epochs': 1
    }
}


# finally, run program
amb = Amber(types=type_dict, specs=specs)
amb.run()
