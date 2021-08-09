"""This module hosts custom operations that are not in standard Keras/Tensorflow library
"""

from .sepfc import SeparableFC
from .denovo_motif_conv import DenovoConvMotif, Layer_deNovo
from .topk import sparsek_vec


__all__ = [
    'SeparableFC',
    'DenovoConvMotif',
    'Layer_deNovo',
    'sparsek_vec'
]