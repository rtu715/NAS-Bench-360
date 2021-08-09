"""Test architect optimizer"""

import tensorflow as tf
import numpy as np
import copy
import tempfile
from amber.utils import testing_utils
from amber import architect


class TestModelSpace(testing_utils.TestCase):
    def test_conv1d_model_space(self):
        model_space = architect.ModelSpace()
        num_layers = 2
        out_filters = 8
        layer_ops = [
                architect.Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu'),
                architect.Operation('conv1d', filters=out_filters, kernel_size=4, activation='relu'),
                architect.Operation('maxpool1d', filters=out_filters, pool_size=4, strides=1),
                architect.Operation('avgpool1d', filters=out_filters, pool_size=4, strides=1),
        ]
        # init
        for i in range(num_layers):
            model_space.add_layer(i, copy.copy(layer_ops))
        self.assertLen(model_space, num_layers)
        self.assertLen(model_space[0], 4)
        # Add layer
        model_space.add_layer(2, copy.copy(layer_ops))
        self.assertLen(model_space, num_layers + 1)
        # Add op
        model_space.add_state(2, architect.Operation('identity', filters=out_filters))
        self.assertLen(model_space[2], 5)
        # Delete op
        model_space.delete_state(2, 4)
        self.assertLen(model_space[2], 4)
        # Delete layer
        model_space.delete_layer(2)
        self.assertLen(model_space, num_layers)


class TestGeneralController(testing_utils.TestCase):
    def setUp(self):
        super(TestGeneralController, self).setUp()
        self.session = tf.Session()
        self.model_space, _ = testing_utils.get_example_conv1d_space()
        self.controller = architect.GeneralController(
            model_space=self.model_space,
            buffer_type='ordinal',
            with_skip_connection=True,
            kl_threshold=0.05,
            buffer_size=15,
            batch_size=5,
            session=self.session,
            train_pi_iter=2,
            lstm_size=32,
            lstm_num_layers=1,
            lstm_keep_prob=1.0,
            optim_algo="adam",
            skip_target=0.8,
            skip_weight=0.4,
        )

    def test_get_architecture(self):
        act, prob = self.controller.get_action()
        self.assertIsInstance(act, np.ndarray)
        self.assertIsInstance(prob, list)
        # initial probas should be close to uniform
        i = 0
        for layer_id in range(len(self.model_space)):
            # operation
            pr = prob[i]
            self.assertAllClose(pr.flatten(), [1./len(pr.flatten())] * len(pr.flatten()), atol=0.05)
            # skip connection
            if layer_id > 0:
                pr = prob[i + 1]
                self.assertAllClose(pr.flatten(), [0.5] * len(pr.flatten()), atol=0.05)
                i += 1
            i += 1

    def test_optimize(self):
        act, prob = self.controller.get_action()
        feed_dict = {self.controller.input_arc[i]: [[act[i]]]
                     for i in range(len(act))}
        feed_dict.update({self.controller.advantage: [[1]]})
        feed_dict.update({self.controller.old_probs[i]: prob[i]
                          for i in range(len(self.controller.old_probs))})
        feed_dict.update({self.controller.reward: [[1]]})
        for _ in range(100):
            self.session.run(self.controller.train_op, feed_dict=feed_dict)
        act2, prob2 = self.controller.get_action()
        self.assertAllEqual(act, act2)


class TestOperationController(testing_utils.TestCase):
    def setUp(self):
        super(TestOperationController, self).setUp()
        self.model_space, _ = testing_utils.get_example_conv1d_space()
        self.controller = architect.OperationController(
            state_space=self.model_space,
            controller_units=8,
            kl_threshold=0.05,
            buffer_size=15,
            batch_size=5,
            train_pi_iter=2
        )
        self.tempdir = tempfile.TemporaryDirectory()

    def test_optimize(self):
        seed = np.array([0]*3).reshape((1, 1, 3))
        for _ in range(8):
            act, proba = self.controller.get_action(seed)
            self.controller.store(
                state=seed,
                prob=proba,
                action=act,
                reward=_
            )
        self.controller.train(episode=0, working_dir=self.tempdir.name)

    def tearDown(self):
        super(TestOperationController, self).tearDown()
        self.tempdir.cleanup()


if __name__ == '__main__':
    tf.test.main()

