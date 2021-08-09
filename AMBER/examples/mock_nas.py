import os
import sys
import logging
from amber.architect.controller import GeneralController
from amber.architect.modelSpace import State, ModelSpace
from amber.architect.trainEnv import ControllerTrainEnvironment
from amber.bootstrap.mock_manager import MockManager
from amber.bootstrap.simple_conv1d_space import get_state_space
from amber.utils.logging import setup_logger
import tensorflow as tf


def get_controller(state_space, sess):
    """Test function for building controller network. A controller is a LSTM cell that predicts the next
    layer given the previous layer and all previous layers (as stored in the hidden cell states). The
    controller model is trained by policy gradients as in reinforcement learning.
    """
    with tf.device("/cpu:0"):
        controller = GeneralController(
            model_space=state_space,
            lstm_size=32,
            lstm_num_layers=1,
            with_skip_connection=False,
            kl_threshold=0.1,
            train_pi_iter=100,
            optim_algo='adam',
            # tanh_constant=1.5,
            buffer_size=5,  ## num of episodes saved
            batch_size=5,
            session=sess,
            use_ppo_loss=True
        )
    return controller


def get_mock_manager(history_fn_list, Lambda=1., wd='./tmp_mock'):
    """Test function for building a mock manager. A mock manager
    returns a loss and knowledge instantly based on previous
    training history.
    Args:
        train_history_fn_list: a list of
    """
    manager = MockManager(
        history_fn_list=history_fn_list,
        model_compile_dict={'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ['acc']},
        working_dir=wd,
        Lambda=Lambda,
        verbose=0
    )
    return manager


def get_environment(controller, manager, should_plot, logger=None, wd='./tmp_mock/'):
    """Test function for getting a training environment for controller.
    Args:
        controller: a built controller net
        manager: a manager is a function that manages child-networks. Manager is built upon `model_fn` and `reward_fn`.
        max_trials: maximum number of child-net generated
        working_dir: specifies the working dir where all files will be generated in.
        resume_prev_run: restore the controller parameters and states from a preivous run
        logger: a Logging file handler; creates a new logger if None.
    """
    env = ControllerTrainEnvironment(
        controller,
        manager,
        max_episode=50,
        max_step_per_ep=3,
        logger=logger,
        resume_prev_run=False,
        should_plot=should_plot,
        working_dir=wd,
        with_skip_connection=False,
        with_input_blocks=False
    )
    return env


def train_simple_controller(should_plot=False, logger=None, Lambda=1., wd='./outputs/mock_nas/'):
    sess = tf.Session()

    # first get state_space
    state_space = get_state_space()

    # init network manager
    hist_file_list = ["./data/mock_black_box/tmp_%i/train_history.csv" % i for i in range(1, 21)]
    # print(hist_file_list)
    manager = get_mock_manager(hist_file_list, Lambda=Lambda, wd=wd)

    # get controller
    controller = get_controller(state_space, sess)

    # get the training environment
    env = get_environment(controller, manager, should_plot, logger, wd=wd)

    # train one step
    env.train()


if __name__ == '__main__':
    wd = "./outputs/mock_nas/"
    os.makedirs(wd, exist_ok=True)
    logger = setup_logger(working_dir=wd, verbose_level=logging.DEBUG)
    B = 1
    Lambda = 1.
    for t in range(B):
        train_simple_controller(
            should_plot=t == (B - 1),
            logger=logger,
            Lambda=Lambda
        )
