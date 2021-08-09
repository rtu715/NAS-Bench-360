# -*- coding: utf8 -*-

"""Read a set of train_history.csv files, and
make a mock manager that returns a list of mock
validation metrics given any child net architecture
"""
import os

import numpy as np

from ..architect import GeneralManager
from ..utils.io import read_history


def get_mock_reward(model_states, train_history_df, metric, stringify_states=True):
    if stringify_states:
        model_states_ = [str(x) for x in model_states]
    else:
        model_states_ = model_states
    idx_bool = np.array([train_history_df['L%i' % (i + 1)] == model_states_[i] for i in range(len(model_states_))])
    index = np.apply_along_axis(func1d=lambda x: all(x), axis=0, arr=idx_bool)
    if np.sum(index) == 0:
        idx = train_history_df['loss'].idxmax()
        return train_history_df[metric].iloc[idx]
    # return train_history_df[metric].iloc[np.random.choice(train_history_df.index)]
    # raise Exception("cannot find config in history: \n {}".format(",".join(str(model_states_[i]) for i in range(len(model_states_)) ) ) )
    else:
        # mu, sd = np.mean(train_history_df[metric].iloc[index]), np.std(train_history_df[metric].iloc[index])
        return train_history_df[metric].iloc[np.random.choice(np.where(index)[0])]


def get_default_mock_reward_fn(model_states, train_history_df, lbd=1.0, metric=['loss', 'knowledge', 'acc']):
    Lambda = lbd
    mock_reward = get_mock_reward(model_states, train_history_df, metric)
    this_reward = -(mock_reward['loss'] + Lambda * mock_reward['knowledge'])
    loss_and_metrics = [mock_reward['loss'], mock_reward['acc']]
    reward_metrics = {'knowledge': mock_reward['knowledge']}
    return this_reward, loss_and_metrics, reward_metrics


def get_mock_reward_fn(train_history_df, metric, stringify_states, lbd=1.):
    def reward_fn(model_states, *args, **kwargs):
        mock_reward = get_mock_reward(model_states, train_history_df, metric, stringify_states)
        this_reward = -(mock_reward['loss'] + lbd * mock_reward['knowledge'])
        loss_and_metrics = [mock_reward['loss']] + [mock_reward[x] for x in metric if x != 'loss' and x != 'knowledge']
        reward_metrics = {'knowledge': mock_reward['knowledge']}
        return this_reward, loss_and_metrics, reward_metrics

    return reward_fn


class MockManager(GeneralManager):
    """Helper class for bootstrapping a random reward for any given architecture from a set of history records"""

    def __init__(self,
                 history_fn_list,
                 model_compile_dict,
                 train_data=None,
                 validation_data=None,
                 input_state=None,
                 output_state=None,
                 model_fn=None,
                 reward_fn=None,
                 post_processing_fn=None,
                 working_dir='.',
                 Lambda=1.,
                 acc_beta=0.8,
                 clip_rewards=0.0,
                 metric_name_dict={'acc': 0, 'knowledge': 1, 'loss': 2},
                 verbose=0):
        # super(MockManager, self).__init__()
        assert type(Lambda) in (float, int), "Lambda potentially confused with `Keras.core.Lambda` layer"
        self._lambda = Lambda
        self.reward_fn = reward_fn
        self.model_fn = model_fn
        self.model_compile_dict = model_compile_dict
        self.train_history_df = read_history(history_fn_list, metric_name_dict)
        self.clip_rewards = clip_rewards
        self.verbose = verbose
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        self.beta = acc_beta
        self.beta_bias = acc_beta
        self.moving_reward = 0.0

    def __str__(self):
        s = "MockManager with %i records" % self.train_history_df.shape[0]
        return s

    def get_rewards(self, trial, model_states=None, **kwargs):
        if model_states is None:
            model_states = kwargs.pop('model_arc', None)
        # evaluate the model by `reward_fn`
        if self.reward_fn:
            this_reward, loss_and_metrics, reward_metrics = self.reward_fn(model_states, self.train_history_df,
                                                                           self._lambda)
        else:
            this_reward, loss_and_metrics, reward_metrics = get_default_mock_reward_fn(model_states,
                                                                                       self.train_history_df,
                                                                                       self._lambda)
        loss = loss_and_metrics.pop(0)
        loss_and_metrics = {str(self.model_compile_dict['metrics'][i]): loss_and_metrics[i] for i in
                            range(len(loss_and_metrics))}
        loss_and_metrics['loss'] = loss
        if reward_metrics:
            loss_and_metrics.update(reward_metrics)

        return this_reward, loss_and_metrics
