from collections import OrderedDict

try:
    import gpustat

    _q = gpustat.GPUStatCollection.new_query().jsonify()
    NUM_GPUS = len(_q['gpus'])
except Exception as e:
    print(e)
    NUM_GPUS = 0

PARAMS_LAYOUT = OrderedDict({
    'Target Model-Basics': OrderedDict({
        'model_basics': {
            'value': '--Model Basics--',
            'wpos': {'row': 0, 'column': 0, 'columnspan': 4, 'sticky': 'ew', 'pady': 10}
        },
        'train_data': {
            'value': "Custom..",
            'lpos': {'row': 1, 'column': 0, 'sticky': 'w'},
            'wpos': {'row': 1, 'column': 1, 'sticky': 'w'}
        },
        'validation_data': {
            'value': "Custom..",
            'lpos': {'row': 1, 'column': 2, 'sticky': 'w'},
            'wpos': {'row': 1, 'column': 3, 'sticky': 'w'}
        },
        'model_space': {
            'value': 'Custom..',
            'lpos': {'row': 2, 'column': 0, 'sticky': 'w'},
            'wpos': {'row': 2, 'column': 1, 'sticky': 'w'}
        },

        'model_builder': {
            'value': ['DAG', 'Enas'],
            'lpos': {'row': 4, 'column': 0, 'sticky': 'w'},
            'wpos': {'row': 4, 'column': 1, 'sticky': 'w'}
        },
        'dag_func': {
            'value': ['InputBlockDAG', 'InputBlockAuxLossDAG', 'EnasAnnDAG'],
            'lpos': {'row': 4, 'column': 2, 'sticky': 'w'},
            'wpos': {'row': 4, 'column': 3, 'sticky': 'w'}
        },

        'model_compile_dict': {
            'value': '--Model Compile--',
            'wpos': {'row': 5, 'column': 0, 'columnspan': 4, 'sticky': 'ew', 'pady': 10}
        },
        'optimizer': {
            'value': ['adam', 'sgd', 'momentum', 'adagrad', 'rmsprop'],
            'lpos': {'row': 7, 'column': 0, 'sticky': 'w'},
            'wpos': {'row': 7, 'column': 1, 'sticky': 'w'},
        },
        'child_loss': {
            'value': ['mse', 'categorical_crossentropy'],
            'lpos': {'row': 8, 'column': 0, 'sticky': 'w'},
            'wpos': {'row': 8, 'column': 1, 'sticky': 'w'},
        },
        'child_metrics': {
            'value': '[35, 2]',
            'lpos': {'row': 9, 'column': 0, 'sticky': 'w'},
            'wpos': {'row': 9, 'column': 1, 'columnspan': 3, 'sticky': 'w'},
        }
    }),
    'Target Model-Interpret': {
        'knowledge_basics': {
            'value': '--Knowledge Function--',
            'wpos': {'row': 0, 'column': 0, 'columnspan': 4, 'sticky': 'new', 'pady': 10}
        },
        'knowledge_fn': {
            'value': ['GraphHierarchyTree', 'GraphHierarchyTreeAuxLoss', 'Motif'],
            'lpos': {'row': 1, 'column': 0, 'sticky': 'w'},
            'wpos': {'row': 1, 'column': 1, 'sticky': 'w'}
        },
        'knowledge_data': {
            'value': 'Custom..',
            'lpos': {'row': 2, 'column': 0, 'sticky': 'w'},
            'wpos': {'row': 2, 'column': 1, 'sticky': 'w'}
        },
        'knowledge_specific_settings': {
            'value': '[20, 5]',
            'default': '{}',
            'lpos': {'row': 3, 'column': 0, 'sticky': 'w'},
            'wpos': {'row': 3, 'column': 1, 'sticky': 'w'}
        },
        'reward_basics': {
            'value': '--Reward--',
            'wpos': {'row': 4, 'column': 0, 'columnspan': 4, 'sticky': 'new', 'pady': 10}
        },
        'reward_fn': {
            'value': ['KnowledgeReward', 'LossReward', 'Mock_Reward'],
            'lpos': {'row': 5, 'column': 0, 'sticky': 'w'},
            'wpos': {'row': 5, 'column': 1, 'sticky': 'w'}
        },
        'knowledge_weight': {
            'value': '[8, 1]',
            'default': '1.0',
            'lpos': {'row': 6, 'column': 0, 'sticky': 'w'},
            'wpos': {'row': 6, 'column': 1, 'sticky': 'w'}
        },
        'knowledge_c': {
            'value': '[8, 1]',
            'default': 'None',
            'lpos': {'row': 7, 'column': 0, 'sticky': 'w'},
            'wpos': {'row': 7, 'column': 1, 'sticky': 'w'}
        },
        'loss_c': {
            'value': '[8, 1]',
            'default': 'None',
            'lpos': {'row': 7, 'column': 2, 'sticky': 'w'},
            'wpos': {'row': 7, 'column': 3, 'sticky': 'w'}
        }

    },
    'Controller-Basics': {
        'controller_basics': {
            'value': '--Controller--',
            'wpos': {'row': 0, 'column': 0, 'columnspan': 4, 'sticky': 'new', 'pady': 10}
        },
        'controller_type':
            {
                'value': ['General'],
                'lpos': {'row': 1, 'column': 0, 'sticky': 'w'},
                'wpos': {'row': 1, 'column': 1, 'sticky': 'w'}
            },
        'lstm_layers':
            {
                'value': ['2', '1', '3', '4', '5'],
                'lpos': {'row': 2, 'column': 0, 'sticky': 'w'},
                'wpos': {'row': 2, 'column': 1, 'sticky': 'w'}
            },
        'lstm_size':
            {
                'value': ['32', '16', '64', '128'],
                'lpos': {'row': 2, 'column': 2, 'sticky': 'w'},
                'wpos': {'row': 2, 'column': 3, 'sticky': 'w'}
            },
        'ctrl_lr':
            {
                'value': '[8,1]',
                'default': '0.001',
                'lpos': {'row': 3, 'column': 0, 'sticky': 'w'},
                'wpos': {'row': 3, 'column': 1, 'sticky': 'w'}
            },
        'ctrl_epoch':
            {
                'value': '[8,1]',
                'default': '10',
                'lpos': {'row': 3, 'column': 2, 'sticky': 'w'},
                'wpos': {'row': 3, 'column': 3, 'sticky': 'w'}
            },
        'ctrl_buffer_size':
            {
                'value': ['1', '5', '10', '20', '30'],
                'lpos': {'row': 4, 'column': 0, 'sticky': 'w'},
                'wpos': {'row': 4, 'column': 1, 'sticky': 'w'}
            },
        'ctrl_batch_size':
            {
                'value': ['2', '5', '10'],
                'lpos': {'row': 4, 'column': 2, 'sticky': 'w'},
                'wpos': {'row': 4, 'column': 3, 'sticky': 'w'}
            },
        'kl_cutoff':
            {
                'value': '[8,1]',
                'default': '0.1',
                'lpos': {'row': 5, 'column': 0, 'sticky': 'w'},
                'wpos': {'row': 5, 'column': 1, 'sticky': 'w'}
            },
        'optim_method':
            {
                'value': ['REINFORCE', 'PPO'],
                'lpos': {'row': 6, 'column': 0, 'sticky': 'w'},
                'wpos': {'row': 6, 'column': 1, 'sticky': 'w'}
            },

        'manager_basics': {
            'value': '--Manager--',
            'wpos': {'row': 7, 'column': 0, 'columnspan': 4, 'sticky': 'ew', 'pady': 10}
        },
        'manager_type':
            {
                'value': ['General', 'Enas'],
                'lpos': {'row': 8, 'column': 0, 'sticky': 'w'},
                'wpos': {'row': 8, 'column': 1, 'sticky': 'w'}
            },
        'child_batch_size':
            {
                'value': '[8,1]',
                'default': '128',
                'lpos': {'row': 9, 'column': 0, 'sticky': 'w'},
                'wpos': {'row': 9, 'column': 1, 'sticky': 'w'}
            },
        'child_epochs':
            {
                'value': '[8,1]',
                'default': '20',
                'lpos': {'row': 9, 'column': 2, 'sticky': 'w'},
                'wpos': {'row': 9, 'column': 3, 'sticky': 'w'}
            },
        'postprocessing_fn':
            {
                'value': ['minimal', 'model_plot', 'general', 'regression'],
                'lpos': {'row': 10, 'column': 0, 'sticky': 'w'},
                'wpos': {'row': 10, 'column': 1, 'sticky': 'w'}
            },
        'manager_verbosity':
            {
                'value': ['0', '1', '2'],
                'lpos': {'row': 11, 'column': 0, 'sticky': 'w'},
                'wpos': {'row': 11, 'column': 1, 'sticky': 'w'}
            },
    },
    'Controller-Environ': {
        'env_basics': {
            'value': '--Train Environ--',
            'wpos': {'row': 0, 'column': 0, 'columnspan': 4, 'sticky': 'new', 'pady': 10}
        },
        'env_type':
            {
                'value': ['ControllerTrainEnv', 'EnasTrainEnv'],
                'lpos': {'row': 1, 'column': 0, 'sticky': 'w'},
                'wpos': {'row': 1, 'column': 1, 'sticky': 'w'}
            },
        'total_steps':
            {
                'value': '[8,1]',
                'default': '100',
                'lpos': {'row': 2, 'column': 0, 'sticky': 'w'},
                'wpos': {'row': 2, 'column': 1, 'sticky': 'w'}
            },
        'samples_per_step':
            {
                'value': '[8,1]',
                'default': '3',
                'lpos': {'row': 2, 'column': 2, 'sticky': 'w'},
                'wpos': {'row': 2, 'column': 3, 'sticky': 'w'}
            },

    }
})

STATUS_BAR_LAYOUT = OrderedDict({
    'statusbar': {
        'value': '--Status--',
        'wpos': {'row': 0, 'column': 0, 'columnspan': 2, 'sticky': 'ew', 'pady': 10}
    },

    'run_status': {
        'value': '{8, 1}',
        'default': 'None',
        'caption': '',
        'lpos': {'row': 1, 'column': 0, 'sticky': 'w'},
        'wpos': {'row': 1, 'column': 1, 'sticky': 'w'}
    },

    'cpu_status': {
        'value': '{8, 1}',
        'default': '0.0%',
        'lpos': {'row': 2, 'column': 0, 'sticky': 'w'},
        'wpos': {'row': 2, 'column': 1, 'sticky': 'w'}
    },
    'ram': {
        'value': '{8, 1}',
        'default': '0.0%',
        'lpos': {'row': 3, 'column': 0, 'sticky': 'w'},
        'wpos': {'row': 3, 'column': 1, 'sticky': 'w'}
    },
    'gpu_status': {
        'value': '[8,%i]' % max(NUM_GPUS, 1),
        'default': 'Unavailable',
        'lpos': {'row': 4, 'column': 0, 'sticky': 'w'},
        'wpos': {'row': 5, 'column': 0, 'columnspan': 2, 'sticky': 'we'}
    },
})
