'''
Author: Badri Adhikari, University of Missouri-St. Louis,  11-13-2019
File: Contains custom loss functions
'''

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
import tensorflow as tf
epsilon = tf.keras.backend.epsilon()
from tensorflow.python.ops import nn

def inv_log_cosh(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    def _logcosh(x):
        return x + nn.softplus(-2. * x) - math_ops.log(2.)
    return K.mean(_logcosh(100.0 / (y_pred + epsilon) - 100.0 / (y_true + epsilon) ), axis=-1)
