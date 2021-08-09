from amber.utils import testing_utils
from amber.architect.controller import GeneralController
from amber.architect.trainEnv import ControllerTrainEnvironment
from amber.bootstrap.mock_manager import MockManager
from amber.bootstrap.simple_conv1d_space import get_state_space
import tensorflow as tf
import numpy as np
import os
import tempfile
import platform
if platform.system() == 'Darwin':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def get_controller(state_space, sess):
    """Test function for building controller network. A controller is a LSTM cell that predicts the next
    layer given the previous layer and all previous layers (as stored in the hidden cell states). The
    controller model is trained by policy gradients as in reinforcement learning.
    """
    with tf.device("/cpu:0"):
        controller = GeneralController(
            model_space=state_space,
            lstm_size=16,
            lstm_num_layers=1,
            with_skip_connection=False,
            kl_threshold=0.05,
            train_pi_iter=50,
            optim_algo='adam',
            # tanh_constant=1.5,
            buffer_size=5,
            batch_size=5,
            session=sess,
            use_ppo_loss=False,
            verbose=0
        )
    return controller


def get_mock_manager(history_fn_list, Lambda=1., wd='./tmp_mock'):
    """Test function for building a mock manager. A mock manager
    returns a loss and knowledge instantly based on previous
    training history.
    """
    manager = MockManager(
        history_fn_list=history_fn_list,
        model_compile_dict={'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ['acc']},
        working_dir=wd,
        Lambda=Lambda,
        verbose=0
    )
    return manager


def get_environment(controller, manager, should_plot=True, logger=None, wd='./tmp_mock/'):
    """Test function for getting a training environment for controller.
    """
    env = ControllerTrainEnvironment(
        controller,
        manager,
        max_episode=100,
        max_step_per_ep=3,
        logger=logger,
        resume_prev_run=False,
        should_plot=should_plot,
        working_dir=wd,
        with_skip_connection=False,
        with_input_blocks=False
    )
    return env


class TestBootstrap(testing_utils.TestCase):
    Lambda = 1

    def __init__(self, *args, **kwargs):
        super(TestBootstrap, self).__init__(*args, **kwargs)
        self.hist_file_list = [os.path.join(os.path.dirname(__file__), "mock_black_box/tmp_%i/train_history.csv.gz" % i)
                               for i in range(1, 21)]
        # first get state_space
        self.state_space = get_state_space()

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.session = tf.Session()
        # init network manager
        self.manager = get_mock_manager(self.hist_file_list, Lambda=self.Lambda, wd=self.tempdir.name)
        # get controller
        self.controller = get_controller(self.state_space, self.session)
        # get the training environment
        self.env = get_environment(self.controller, self.manager, wd=self.tempdir.name)

    def _sample_rewards(self):
        rewards = []
        for _ in range(10):
            arc = self.controller.get_action()[0]
            arc = [self.controller.model_space[layer_id][i] for layer_id, i in enumerate(arc) ]
            res = self.manager.get_rewards(-1, model_arc=arc)[0]
            rewards.append(res)
        return rewards

    def test_run(self):
        old_rewards = self._sample_rewards()
        self.env.train()
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir.name, 'train_history.csv')))
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir.name, 'controller_weights.h5')))
        new_rewards = self._sample_rewards()
        self.assertLess(np.mean(old_rewards), np.mean(new_rewards))

    def tearDown(self):
        self.tempdir.cleanup()
        super(TestBootstrap, self).tearDown()


if __name__ == '__main__':
    tf.test.main()

