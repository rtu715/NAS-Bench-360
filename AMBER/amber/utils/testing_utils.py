"""
Testing utilities for amber
"""

import tensorflow as tf
try:
    import keras
except ImportError:
    from tensorflow import keras as keras
from .. import architect


class TestCase(tf.test.TestCase):
    def tearDown(self):
        tf.keras.backend.clear_session()
        keras.backend.clear_session()
        super(TestCase, self).tearDown()


class PseudoModel:
    def __init__(self, pred_retr, eval_retr):
        self.pred_retr = pred_retr
        self.eval_retr = eval_retr

    def predict(self, *args, **kwargs):
        return self.pred_retr

    def evaluate(self, *args, **kwargs):
        return self.eval_retr

    def fit(self, *args, **kwargs):
        pass

    def load_weights(self, *args, **kwargs):
        pass


class PseudoKnowledge:
    def __init__(self, k_val):
        self.k_val = k_val

    def __call__(self, *args, **kwargs):
        return self.k_val


class PseudoConv1dModelBuilder:
    def __init__(self, input_shape, output_units, model_compile_dict=None):
        self.input_shape = input_shape
        self.output_units = output_units
        self.model_compile_dict = model_compile_dict or {'optimizer':'sgd', 'loss':'mse'}
        self.session = None

    def __call__(self, *args, **kwargs):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv1D(filters=4, kernel_size=1, input_shape=self.input_shape))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=self.output_units))
        model.compile(**self.model_compile_dict)
        return model


class PseudoCaller:
    def __init__(self, retr_val=0):
        self.retr_val = retr_val

    def __call__(self, *args, **kwargs):
        return self.retr_val


class PseudoReward(PseudoCaller):
    def __init__(self, retr_val=0):
        self.retr_val = retr_val
        self.knowledge_function = None

    def __call__(self, *args, **kwargs):
        return self.retr_val, [self.retr_val], None


def get_example_conv1d_space(out_filters=8, num_layers=2):
    model_space = architect.ModelSpace()
    num_pool = 1
    expand_layers = [num_layers//k-1 for k in range(1, num_pool)]
    layer_sharing = {}
    for i in range(num_layers):
        model_space.add_layer(i, [
            architect.Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu'),
            architect.Operation('maxpool1d', filters=out_filters, pool_size=4, strides=1),
            architect.Operation('identity', filters=out_filters)
      ])
        if i in expand_layers:
            out_filters *= 2
        if i > 0:
            layer_sharing[i] = 0
    return model_space, layer_sharing

