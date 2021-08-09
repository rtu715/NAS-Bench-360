"""
This module's documentation is Work-In-Progress.

This module might be deprecated in the future.
"""

import matplotlib
from ..architect import OperationController, ControllerTrainEnvironment
from .gold_standard import *
from .mock_manager import *
from .simple_conv1d_space import get_state_space

import matplotlib.pyplot as plt
import json

matplotlib.use('Agg')
working_dir = './tmp_mock/'


def get_controller(state_space):
    """Test function for building controller network. A controller is a LSTM cell that predicts the next

    layer given the previous layer and all previous layers (as stored in the hidden cell states). The
    controller model is trained by policy gradients as in reinforcement learning.

    Parameters
    ----------
    state_space :
        the State_Space object to search optimal layer compositions from.
    """

    controller = OperationController(
        state_space,
        controller_units=50,
        embedding_dim=8,
        optimizer='adam',
        discount_factor=0.0,
        clip_val=0.2,
        kl_threshold=0.01,
        train_pi_iter=100,
        lr_pi=0.1,
        buffer_size=15,  ## num of episodes saved
        batch_size=5
    )
    return controller


def get_mock_manager(history_fn_list, Lambda):
    """Test function for building a mock manager. A mock manager
    returns a loss and knowledge instantly based on previous
    training history.

    Parameters
    ----------
    history_fn_list:
        a list of file path of train history

    Lambda :
        weight
    """
    manager = MockManager(
        history_fn_list=history_fn_list,
        model_compile_dict={'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ['acc']},
        Lambda=Lambda,
        working_dir='./tmp_mock',
        verbose=0
    )
    return manager


def get_environment(controller, manager, should_plot):
    """Test function for getting a training environment for controller.

    Parameters
    ----------
    controller:
        a built controller net

    manager:
        a manager is a function that manages child-networks. Manager is built upon `model_fn` and `reward_fn`.

    should_plot:
        whether plot at the end of controller training
    """
    env = ControllerTrainEnvironment(
        controller,
        manager,
        max_episode=200,
        max_step_per_ep=1,
        logger=None,
        resume_prev_run=False,
        should_plot=should_plot,
        working_dir=working_dir
    )
    return env


def train_simple_controller(should_plot=False, Lambda=1.):
    # first get state_space
    state_space = get_state_space()

    hist_file_list = ["BioNAS/resources/mock_black_box/tmp_%i/train_history.csv" % i for i in range(1, 21)]

    manager = get_mock_manager(hist_file_list, Lambda)

    # get controller
    controller = get_controller(state_space)

    # get the training environment
    env = get_environment(controller, manager, should_plot)

    # train one step
    idx = env.train()

    converged_config_str = []
    for sp, i in zip(state_space, idx):
        converged_config_str.append(str(sp[i]))
        print(str(sp[i]))

    return converged_config_str


def run(lambda_list=[0.01, 0.1, 1., 10., 100.]):
    save_path = os.path.join(working_dir, 'metrics_vs_lambda.json')

    if os.path.exists(save_path):
        df = json.load(open(save_path))
    else:
        df = {'k_rank': [], 'l_rank': [], 'acc': [], 'loss': [], 'knowledge': []}

    hist_file_list = ["BioNAS/resources/mock_black_box/tmp_%i/train_history.csv" % i for i in range(1, 21)]

    nas_file_list = [working_dir + "train_history.csv"]

    metric_list = {key: [] for key in df}
    for Lambda in lambda_list:
        nas_model_str = train_simple_controller(Lambda=Lambda)

        metrics = get_rating(hist_file_list, nas_model_str)

        for metric, value in metrics.items():
            metric_list[metric].append(value)

    for key in df:
        df[key].append(metric_list[key])

    json.dump(df, open(save_path, 'w'))


def plot(lambda_list):
    save_path = os.path.join(working_dir, 'metrics_vs_lambda.json')

    if os.path.exists(save_path):
        df = json.load(open(save_path))
    else:
        raise IOError('file does not exist')

    plt.close()
    for key in df:
        if key != 'k_rank' and key != 'l_rank':
            d = np.array(df[key])
            d = np.sort(d, axis=0)
            min_, max_, mid_ = d[int(d.shape[0] * 0.1), :], d[int(d.shape[0] * 0.9), :], d[int(d.shape[0] * 0.5), :]
            plt.plot(lambda_list, mid_, label=key, marker='o')
            plt.fill_between(lambda_list, min_, max_, alpha=0.2)

    plt.legend(loc='upper left')
    plt.xscale('log', basex=10)
    plt.xlabel('Lambda Value', fontsize=16)
    plt.ylabel('Metrics', fontsize=16)
    plt.title('Metrics of Best Config Found vs. Knowledge Weight')
    plt.savefig(os.path.join(working_dir, 'best_metrics_vs_lambda.pdf'))

    ## plot rank

    plt.close()
    for key in ['k_rank', 'l_rank']:
        d = np.array(df[key])
        d = np.sort(d, axis=0)
        min_, max_, mid_ = d[int(d.shape[0] * 0.1), :], d[int(d.shape[0] * 0.9), :], d[int(d.shape[0] * 0.5), :]
        if key == 'k_rank':
            label = 'knowledge rank'
        else:
            label = 'loss rank'
        plt.plot(lambda_list, mid_, label=label, marker='o')
        plt.fill_between(lambda_list, min_, max_, alpha=0.2)
    plt.legend(loc='upper left')
    plt.xscale('log', basex=10)
    plt.xlabel('Lambda Value', fontsize=16)
    plt.ylabel('Rank', fontsize=16)
    plt.title('Rank of Best Config Found vs. Knowledge Weight')
    plt.savefig(os.path.join(working_dir, 'best_rank_vs_lambda.pdf'))


def main():
    lambda_list = [0.01, 0.1, 1., 10., 100]
    for i in range(100):
        run(lambda_list)

    plot(lambda_list)


if __name__ == '__main__':
    main()
