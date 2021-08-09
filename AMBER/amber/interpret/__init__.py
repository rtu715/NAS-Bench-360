"""
:mod:`interpret` hosts interpretation methods for a model.

Currently only a `Selene <https://selene.flatironinstitute.org/>`_ derived variant effect prediction is used.

In the future, this module will also interact with :mod:`objective` to provide additional insights for a trained model/
architecture.

This module's documentation is Work-In-Progress.
"""

from .sequenceModel import AnalyzeSequencesNAS
from .scores import PrecisionAtRecall, TprAtFpr
from . import heritability

__all__ = [
    'AnalyzeSequencesNAS',
    'PrecisionAtRecall',
    'TprAtFpr',
    'heritability'
]