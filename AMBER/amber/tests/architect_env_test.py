"""Testing train environment test for architects
This is different from other helper tests, because train environment's work depend on helpers being functioning in the
expected behaviors
"""

import os
import tensorflow as tf
import numpy as np
import tempfile
from parameterized import parameterized_class
from amber.utils import testing_utils
from amber import modeler
from amber import architect


def get_random_data(num_samps=1000):
    x = np.random.sample(10*4*num_samps).reshape((num_samps, 10, 4))
    y = np.random.sample(num_samps)
    return x, y


def get_class_name(*args):
    id = args[1]
    map = {
        0 : 'TestGeneralEnv',
        1 : 'TestEnasEnv'
    }
    return map[id]


@parameterized_class(attrs=('foo', 'manager_getter', 'controller_getter', 'modeler_getter', 'trainenv_getter'), input_values=[
    (0, architect.GeneralManager, architect.GeneralController, modeler.KerasResidualCnnBuilder, architect.ControllerTrainEnvironment),
    (1, architect.EnasManager, architect.GeneralController, modeler.EnasCnnModelBuilder, architect.EnasTrainEnv)
], class_name_func=get_class_name)
class TestEnvDryRun(testing_utils.TestCase):
    """Test dry-run will only aim to construct a train env class w/o examining its behaviors; however, this will
    serve as the scaffold for other tests
    """
    manager_getter = architect.GeneralManager
    controller_getter = architect.GeneralController
    modeler_getter = modeler.KerasResidualCnnBuilder
    trainenv_getter = architect.ControllerTrainEnvironment

    def __init__(self, *args, **kwargs):
        super(TestEnvDryRun, self).__init__(*args, **kwargs)
        self.train_data = get_random_data(50)
        self.val_data = get_random_data(10)
        self.model_space, _ = testing_utils.get_example_conv1d_space(out_filters=8, num_layers=2)
        self.reward_fn = architect.reward.LossReward()
        self.store_fn = architect.store.get_store_fn('minimal')

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.session = tf.Session()
        self.controller = self.controller_getter(
            model_space=self.model_space,
            buffer_type='ordinal',
            with_skip_connection=True,
            kl_threshold=0.05,
            buffer_size=1,
            batch_size=3,
            session=self.session,
            train_pi_iter=10,
            lstm_size=16,
            lstm_num_layers=1,
            optim_algo="adam",
            skip_target=0.8,
            skip_weight=0.4,
        )
        self.model_fn = self.modeler_getter(
            model_space=self.model_space,
            inputs_op=architect.Operation('input', shape=(10, 4)),
            output_op=architect.Operation('dense', units=1, activation='sigmoid'),
            fc_units=5,
            flatten_mode='gap',
            model_compile_dict={'optimizer': 'adam', 'loss': 'mse'},
            batch_size=10,
            session=self.session,
            controller=self.controller,
            verbose=0
        )
        self.manager = self.manager_getter(
            train_data=self.train_data,
            validation_data=self.val_data,
            model_fn=self.model_fn,
            reward_fn=self.reward_fn,
            store_fn=self.store_fn,
            working_dir=self.tempdir.name,
            child_batchsize=10,
            epochs=1,
            verbose=0
        )
        self.env = self.trainenv_getter(
            self.controller,
            self.manager,
            max_episode=20,
            max_step_per_ep=1,
            logger=None,
            resume_prev_run=False,
            should_plot=True,
            working_dir=self.tempdir.name,
            with_skip_connection=True
        )

    def tearDown(self):
        super(TestEnvDryRun, self).tearDown()
        self.tempdir.cleanup()

    def test_build(self):
        self.env.train()
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir.name, 'controller_weights.h5')))
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir.name, 'train_history.csv')))


if __name__ == '__main__':
    tf.test.main()
