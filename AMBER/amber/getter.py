# -*- coding: utf-8 -*-

"""
This script wraps around the Controller and returns a working
TrainEnv from a configuration file


Author : zzjfrank
Date   : 11.6.2019
"""

import os


class DataToParse:
    def __init__(self, path, method=None):
        self.path = path
        self.method = method
        self._extension()

    def _extension(self):
        ext = os.path.splitext(self.path)[1]
        if ext in ('.pkl', '.pickle'):
            self.method = 'pickle'
        elif ext in ('.npy',):
            self.method = 'numpy'
        elif ext in ('.h5', '.hdf5'):
            self.method = 'hdf5'
        else:
            raise Exception("Unknown data format: %s"% self.path)

    def __str__(self):
        s = "DataToParse-%s" % self.path
        return s

    def unpack(self):
        assert os.path.isfile(self.path), "File does not exist: %s" % self.path
        assert self.method is not None, "Cannot determine parse method for file: %s" % self.path
        print("unpacking data.. %s" % self.path)
        if self.method == 'pickle':
            import pickle
            return pickle.load(open(self.path, 'rb'))
        elif self.method == 'numpy':
            import numpy
            return numpy.load(self.path)
        elif self.method == 'hdf5':
            import h5py
            return h5py.File(self.path, 'r')


def load_data_dict(d):
    for k, v in d.items():
        if type(v) is DataToParse:
            d[k] = v.unpack()
        elif type(v) is str:
            assert os.path.isfile(v), "cannot find file: %s" % v
            d[k] = DataToParse(v).unpack()
    return d


# this is the ultimate goal; needs controller and manager
def get_train_env(env_type, controller, manager, *args, **kwargs):
    if env_type == 'ControllerTrainEnv':
        from .architect.trainEnv import ControllerTrainEnvironment
        env = ControllerTrainEnvironment(controller=controller, manager=manager,
                                         *args, **kwargs)
    elif env_type == 'EnasTrainEnv':
        from .architect.trainEnv import EnasTrainEnv
        env = EnasTrainEnv(controller=controller, manager=manager,
                           *args, **kwargs)
    else:
        raise Exception("cannot understand manager type: %s" % env_type)
    print("env_type = %s" % env_type)
    return env


# controller; needs model_space
def get_controller(controller_type, model_space, session, **kwargs):
    if controller_type == 'General' or controller_type == 'GeneralController':
        from .architect import GeneralController
        controller = GeneralController(model_space=model_space, session=session, **kwargs)
    elif controller_type == 'Operation' or controller_type == 'OperationController':
        from .architect import OperationController
        controller = OperationController(model_space=model_space, **kwargs)
    elif controller_type == 'MultiIO' or controller_type == 'MultiIOController':
        from .architect import MultiIOController
        controller = MultiIOController(model_space=model_space, session=session, **kwargs)
    elif controller_type == 'ZeroShot' or controller_type == 'ZeroShotController':
        from .architect import ZeroShotController
        controller = ZeroShotController(model_space=model_space, session=session, **kwargs)
    else:
        raise Exception('cannot understand controller type: %s' % controller_type)
    print("controller = %s" % controller_type)
    return controller


# model_space
def get_model_space(arg):
    from .architect.modelSpace import ModelSpace
    if type(arg) is str:
        if arg == 'Default ANN':
            from .bootstrap.dense_skipcon_space import get_model_space as ms_ann
            model_space = ms_ann(3)
        elif arg == 'Default 1D-CNN':
            from .bootstrap.simple_conv1d_space import get_state_space as ms_cnn
            model_space = ms_cnn()
        else:
            raise Exception("cannot understand string model_space arg: %s" % arg)
    elif type(arg) in (dict, list):
        model_space = ModelSpace.from_dict(arg)
    elif isinstance(arg, ModelSpace):
        model_space = arg
    else:
        raise Exception("cannot understand non-string model_space arg: %s" % arg)
    return model_space


# manager; needs data, model_fn, reward_fn
def get_manager(manager_type, model_fn, reward_fn, data_dict, session, *args, **kwargs):
    data_dict = load_data_dict(data_dict)
    if manager_type == 'General' or manager_type == 'GeneralManager':
        from .architect.manager import GeneralManager
        manager = GeneralManager(model_fn=model_fn,
                                 reward_fn=reward_fn,
                                 train_data=data_dict['train_data'],
                                 validation_data=data_dict['validation_data'],
                                 *args,
                                 **kwargs
                                 )
    elif manager_type == 'EnasManager' or manager_type == 'Enas':
        from .architect.manager import EnasManager
        manager = EnasManager(
            model_fn=model_fn,
            reward_fn=reward_fn,
            train_data=data_dict['train_data'],
            validation_data=data_dict['validation_data'],
            session=session,
            *args,
            **kwargs
        )
    elif manager_type == 'Mock' or manager_type == 'MockManager':
        from .bootstrap.mock_manager import MockManager
        manager = MockManager(
            model_fn=model_fn,
            reward_fn=reward_fn,
            *args,
            **kwargs
        )
    elif manager_type == 'Distributed' or manager_type == 'DistributedManager':
        from .architect.manager import DistributedGeneralManager
        train_data_kwargs = kwargs.pop("train_data_kwargs", None)
        validate_data_kwargs = kwargs.pop("validate_data_kwargs", None)
        devices = kwargs.pop("devices", None)
        manager = DistributedGeneralManager(
                                 devices=devices,
                                 train_data_kwargs=train_data_kwargs,
                                 validate_data_kwargs=validate_data_kwargs,
                                 model_fn=model_fn,
                                 reward_fn=reward_fn,
                                 train_data=data_dict['train_data'],
                                 validation_data=data_dict['validation_data'],
                                 *args,
                                 **kwargs
                                 )

    else:
        raise Exception("cannot understand manager type: %s" % manager_type)
    print("manager = %s" % manager_type)
    return manager


# model_fn
def get_modeler(model_fn_type, model_space, session, *args, **kwargs):
    from .architect.modelSpace import State
    if model_fn_type == 'DAG' or model_fn_type == 'DAGModelBuilder':
        from .modeler import DAGModelBuilder
        assert 'inputs_op' in kwargs and 'outputs_op' in kwargs
        inp_op_list = kwargs.pop("inputs_op")
        inputs_op = [State(**x) if not isinstance(x, State) else x for x in inp_op_list]
        out_op_list = kwargs.pop("outputs_op")
        output_op = [State(**x) if not isinstance(x, State) else x for x in out_op_list]
        model_fn = DAGModelBuilder(
            model_space=model_space,
            num_layers=len(model_space),
            inputs_op=inputs_op,
            output_op=output_op,
            session=session,
            *args, **kwargs)
    elif model_fn_type == 'Enas' or model_fn_type == 'EnasAnnModelBuilder':
        from .modeler import EnasAnnModelBuilder
        inp_op_list = kwargs.pop("inputs_op")
        inputs_op = [State(**x) if not isinstance(x, State) else x for x in inp_op_list]
        out_op_list = kwargs.pop("outputs_op")
        output_op = [State(**x) if not isinstance(x, State) else x for x in out_op_list]
        model_fn = EnasAnnModelBuilder(
            model_space=model_space,
            num_layers=len(model_space),
            inputs_op=inputs_op,
            output_op=output_op,
            session=session,
            *args, **kwargs)
    elif model_fn_type == 'EnasCnnModelBuilder':
        from .modeler import EnasCnnModelBuilder
        inp_op_list = kwargs.pop("inputs_op")
        inputs_op = [State(**x) if not isinstance(x, State) else x for x in inp_op_list]
        out_op_list = kwargs.pop("outputs_op")
        output_op = [State(**x) if not isinstance(x, State) else x for x in out_op_list]
        controller = kwargs.pop('controller')
        model_fn = EnasCnnModelBuilder(
            model_space=model_space,
            num_layers=len(model_space),
            inputs_op=inputs_op,
            output_op=output_op,
            session=session,
            controller=controller,
            *args, **kwargs)
    elif model_fn_type == "KerasModelBuilder":
        from .modeler import KerasModelBuilder
        inp_op_list = kwargs.pop("inputs_op")
        inputs_op = [State(**x) if not isinstance(x, State) else x for x in inp_op_list]
        assert len(inputs_op)==1, "KerasModelBuilder only accepts one input; try KerasMultiIOModelBuilder for multiple inputs"
        out_op_list = kwargs.pop("outputs_op")
        output_op = [State(**x) if not isinstance(x, State) else x for x in out_op_list]
        assert len(output_op)==1, "KerasModelBuilder only accepts one output; try KerasMultiIOModelBuilder for multiple outputs"
        model_fn = KerasModelBuilder(
                inputs=inputs_op[0],
                outputs=output_op[0],
                model_space=model_space,
                *args, **kwargs)
    elif model_fn_type == 'KerasMultiIOModelBuilder':
        from .modeler import KerasMultiIOModelBuilder
        inp_op_list = kwargs.pop("inputs_op")
        inputs_op = [State(**x) if not isinstance(x, State) else x for x in inp_op_list]
        out_op_list = kwargs.pop("outputs_op")
        output_op = [State(**x) if not isinstance(x, State) else x for x in out_op_list]
        model_fn = KerasMultiIOModelBuilder(
            model_space=model_space,
            inputs_op=inputs_op,
            output_op=output_op,
            session=session,
            *args, **kwargs)

    elif model_fn_type == 'KerasBranchModelBuilder':
        from .modeler import KerasBranchModelBuilder
        inp_op_list = kwargs.pop("inputs_op")
        inputs_op = [State(**x) if not isinstance(x, State) else x for x in inp_op_list]
        out_op_list = kwargs.pop("outputs_op")
        output_op = [State(**x) if not isinstance(x, State) else x for x in out_op_list]
        assert len(output_op) == 1
        model_fn = KerasBranchModelBuilder(
            model_space=model_space,
            inputs_op=inputs_op,
            output_op=output_op[0],
            *args, **kwargs) 

    else:
        raise Exception('cannot understand model_builder type: %s' % model_fn_type)
    print("modeler = %s" % model_fn_type)
    return model_fn


# reward_fn; depends on knowledge function
def get_reward_fn(reward_fn_type, knowledge_fn, *args, **kwargs):
    if reward_fn_type == 'KnowledgeReward':
        from .architect.reward import KnowledgeReward
        reward_fn = KnowledgeReward(knowledge_fn, *args, **kwargs)
    elif reward_fn_type == 'LossReward':
        from .architect.reward import LossReward
        assert knowledge_fn is None, "Incompatability: LossReward must have knownledge_fn=None; got %s" % knowledge_fn
        reward_fn = LossReward(*args, **kwargs)
    elif reward_fn_type == 'Mock_Reward':
        from .architect.reward import MockReward
        reward_fn = MockReward(*args, **kwargs)
    elif reward_fn_type == 'LossAucReward':
        from .architect.reward import LossAucReward
        #assert knowledge_fn is None, \
        #    "Incompatability: LossAucReward must have knownledge_fn=None; got %s" % knowledge_fn
        reward_fn = LossAucReward(knowledge_function=knowledge_fn, *args, **kwargs)
    else:
        raise Exception("cannot understand reward_fn type: %s" % reward_fn_type)
    print("reward = %s" % reward_fn_type)
    return reward_fn


# knowledge_fn
def get_knowledge_fn(knowledge_fn_type, knowledge_data_dict, *args, **kwargs):
    if knowledge_data_dict is not None:
        knowledge_data_dict = load_data_dict(knowledge_data_dict)
    if knowledge_fn_type == 'ght' or knowledge_fn_type == 'GraphHierarchyTree':
        from .objective import GraphHierarchyTree
        k_fn = GraphHierarchyTree(*args, **kwargs)
    elif knowledge_fn_type == 'ghtal' or knowledge_fn_type == 'GraphHierarchyTreeAuxLoss':
        from .objective import GraphHierarchyTreeAuxLoss
        k_fn = GraphHierarchyTreeAuxLoss(*args, **kwargs)
    elif knowledge_fn_type == 'Motif':
        from .objective import MotifKLDivergence
        k_fn = MotifKLDivergence(*args, **kwargs)
    elif knowledge_fn_type == 'AuxilaryAcc':
        from .objective import AuxilaryAcc
        k_fn = AuxilaryAcc(*args, **kwargs)
    elif knowledge_fn_type == 'None' or knowledge_fn_type == 'zero':
        k_fn = None
    else:
        raise Exception("cannot understand knowledge_fn type: %s" % knowledge_fn_type)
    if k_fn is not None:
        if hasattr(k_fn, "knowledge_encoder"):
            k_fn.knowledge_encoder(**knowledge_data_dict)
    print("knowledge = %s" % knowledge_fn_type)
    return k_fn


def get_model_and_io_nodes(model_space_arg):
    import json
    import ast

    def eval_shape(d_):
        for j in range(len(d_)):
            if 'shape' in d_[j] and type(d_[j]['shape']) is str:
                d_[j]['shape'] = ast.literal_eval(d_[j]['shape'])
        return d_

    if os.path.isfile(model_space_arg):
        with open(model_space_arg, 'r') as f:
            d = json.load(f)
        # model_space = get_model_space(d['model_space'])
        model_space = d['model_space']
        d['input_states'] = eval_shape(d['input_states'])
        input_states = d['input_states']
        d['output_state'] = eval_shape([d['output_state']])[0]
        output_state = d['output_state']
        return model_space, input_states, output_state
    else:
        raise Exception("cannot open file: %s" % model_space_arg)


# mapping gui var_dict to bionas
def gui_mapper(var_dict):
    # import ast
    wd = var_dict['wd']
    train_data = DataToParse(var_dict['train_data'])
    val_data = DataToParse(var_dict['validation_data'])
    model_space, input_states, output_state = get_model_and_io_nodes(var_dict['model_space'])
    model_compile_dict = {'optimizer': var_dict['optimizer'], 'loss': var_dict['child_loss'],
                          'metrics': [x.strip() for x in var_dict['child_metrics'].strip('[]').split(',')
                                      if len(x.strip())]}
    # this creates a safety issue...
    # might just work for now.. ZZJ 11.9.2019
    # knowledge_params = ast.literal_eval(var_dict['knowledge_specific_settings'])
    knowledge_params = eval(var_dict['knowledge_specific_settings'])
    assert type(knowledge_params) is dict, "Error in parsing `knowledge_specific settings`, must be a dict:\n " \
                                           "%s" % knowledge_params
    knowledge_data = DataToParse(var_dict['knowledge_data']).unpack()

    type_dict = {
        'controller_type': var_dict['controller_type'],
        'model_fn_type': var_dict['model_builder'],
        'knowledge_fn_type': var_dict['knowledge_fn'],
        'reward_fn_type': var_dict['reward_fn'],
        'manager_type': var_dict['manager_type'],
        'env_type': var_dict['env_type']
    }

    specs = {
        'controller':
            {
                'use_ppo_loss': var_dict['optim_method'] == 'PPO',
                # 'with_skip_connection': True,
                # 'with_input_blocks': True,
                'num_input_blocks': len(input_states),
                # 'input_block_unique_connection': True,
                'lstm_size': int(var_dict['lstm_size']),
                'lstm_num_layers': int(var_dict['lstm_layers']),
                'kl_threshold': float(var_dict['kl_cutoff']),
                'train_pi_iter': int(var_dict['ctrl_epoch']),
                # 'skip_weight': None,
                'lr_init': float(var_dict['ctrl_lr']),
                'buffer_size': int(var_dict['ctrl_buffer_size']),
                'batch_size': int(var_dict['ctrl_batch_size'])
            },
        'model_space': model_space,
        'model_builder':
            {
                'input_states': input_states,
                'output_state': output_state,
                # 'with_input_blocks': True,
                # 'with_skip_connection': True,
                'model_compile_dict': model_compile_dict,
                'dag_func': var_dict['dag_func']
            },

        'knowledge_fn':
            {
                'params': knowledge_params,
                'data': knowledge_data
            },

        'reward_fn':
            {
                'Lambda': float(var_dict['knowledge_weight']),
                'knowledge_c': None if var_dict['knowledge_c'] == 'None' else float(var_dict['knowledge_c']),
                'loss_c': None if var_dict['loss_c'] == 'None' else float(var_dict['loss_c']),
            },

        'manager':
            {
                'params': {
                    'working_dir': wd,
                    'model_compile_dict': model_compile_dict,
                    'post_processing_fn': var_dict['postprocessing_fn'],
                    'epochs': int(var_dict['child_epochs']),
                    'verbose': int(var_dict['manager_verbosity']),
                    'child_batchsize': int(var_dict['child_batch_size']),
                },
                'data': {
                    'train_data': train_data,
                    'validation_data': val_data,
                }
            },

        'train_env':
            {
                'max_episode': int(var_dict['total_steps']),
                'max_step_per_ep': int(var_dict['samples_per_step']),
                # 'with_input_blocks': True,
                # 'with_skip_connection': True,
                # 'logger': None,
                # 'resume_prev_run': False,
                'should_plot': True,
                'working_dir': wd,
                'squeezed_action': True,
                # 'save_controller': False,
                # 'continuous_run': False
            }
    }

    return type_dict, specs
