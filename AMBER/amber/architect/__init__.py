"""
The :mod:`architect` module provides neural architecture search implementations and its related helpers

"""


from .controller import GeneralController, MultiInputController, MultiIOController, OperationController
from .modelSpace import State, ModelSpace
from .manager import GeneralManager, EnasManager
from .trainEnv import ControllerTrainEnvironment, EnasTrainEnv
from . import buffer, store, reward, trainEnv, modelSpace, controller

# alias
Operation = State

# TODO: Do not include MultiIO until its tested in multiio branch
__all__ = [
    # funcs
    'GeneralController',
    'Operation',
    'State',
    'ModelSpace',
    'GeneralManager',
    'EnasManager',
    'ControllerTrainEnvironment',
    'EnasTrainEnv',
    # modules
    'buffer',
    'store',
    'reward',
    'trainEnv',
    'modelSpace',
    'controller',
    # For legacy use
    'OperationController',
    'NetworkManager'
]