"""
Modeler is an interface class that interacts outside with manager, and inside coordinates with dag and child.
- Dag builds the underlying tensors
- child facilitates training and evaluating
"""

from .enasModeler import DAGModelBuilder, EnasAnnModelBuilder, EnasCnnModelBuilder
from .kerasModeler import KerasModelBuilder, KerasMultiIOModelBuilder, KerasResidualCnnBuilder, \
    build_sequential_model, build_multi_gpu_sequential_model, \
    build_multi_gpu_sequential_model_from_string, build_sequential_model_from_string, \
    KerasBranchModelBuilder


__all__ = [
    'DAGModelBuilder',
    'EnasCnnModelBuilder',
    'EnasAnnModelBuilder',
    'KerasModelBuilder',
    'KerasMultiIOModelBuilder',
    'KerasResidualCnnBuilder',
    'KerasBranchModelBuilder'
    #'build_sequential_model',
    #'build_sequential_model_from_string',
    #'build_multi_gpu_sequential_model_from_string',
    #'build_multi_gpu_sequential_model'
]
