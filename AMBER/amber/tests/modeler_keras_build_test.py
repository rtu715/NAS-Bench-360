"""Test Keras modeler"""

import tensorflow as tf
import numpy as np
from parameterized import parameterized

from amber.utils import testing_utils
from amber import modeler
from amber import architect


class TestKerasBuilder(testing_utils.TestCase):
    def setUp(self):
        self.model_space, _ = testing_utils.get_example_conv1d_space(num_layers=2)
        self.target_arc = [0, 0, 1]
        self.input_op = architect.Operation('input', shape=(10, 4), name="input")
        self.output_op = architect.Operation('dense', units=1, activation='sigmoid', name="output")
        self.model_compile_dict = {'loss': 'binary_crossentropy', 'optimizer': 'sgd'}
        self.x = np.random.choice(2, 40).reshape((1, 10, 4))
        self.y = np.random.sample(1).reshape((1, 1))
        self.modeler = modeler.KerasResidualCnnBuilder(
            inputs_op=self.input_op,
            output_op=self.output_op,
            model_space=self.model_space,
            fc_units=5,
            flatten_mode='flatten',
            model_compile_dict=self.model_compile_dict
        )

    def test_get_model(self):
        model = self.modeler(self.target_arc)
        old_loss = model.evaluate(self.x, self.y)
        model.fit(self.x, self.y, batch_size=1, epochs=100, verbose=0)
        new_loss = model.evaluate(self.x, self.y)
        self.assertLess(new_loss, old_loss)


class TestKerasGetLayer(testing_utils.TestCase):

    @parameterized.expand([
        # fc
        ((100,), 'dense', {'units': 4}),
        ((100,), 'identity', {}),
        # 1d
        ((100, 4), 'conv1d', {'filters': 5, 'kernel_size': 8}),
        # ((100, 4), 'denovo', {'filters': 5, 'kernel_size': 8}),
        ((100, 4), 'maxpool1d', {'pool_size': 4, 'strides': 4}),
        ((100, 4), 'avgpool1d', {'pool_size': 4, 'strides': 4}),
        ((100, 4), 'lstm', {'units': 2}),
        # reshape
        ((100, 4), 'flatten', {}),
        ((100, 4), 'globalavgpool1d', {}),
        ((100, 4), 'globalmaxpool1d', {}),
        # ((100, 4), 'sfc', {'output_dim': 100, 'symmetric': False}),
        # regularizer
        ((100,), 'dropout', {'rate': 0.3}),
        ((100, 4), 'dropout', {'rate': 0.3}),
        ((100,), 'sparsek_vec', {}),
        ((100,), 'gaussian_noise', {'stddev': 1}),

    ])
    def test_get_layers(self, input_shape, layer_name, layer_attr):
        x = modeler.dag.get_layer(x=None, state=architect.Operation('Input', shape=input_shape))
        operation = architect.Operation(layer_name, **layer_attr)
        layer = modeler.dag.get_layer(x=x, state=operation)
        self.assertIsInstance(layer, tf.Tensor)


if __name__ == '__main__':
    tf.test.main()
