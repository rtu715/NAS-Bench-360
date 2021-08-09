"""Test architect helpers (that is, buffer, reward, store, manager)
"""

import os
import unittest
from parameterized import parameterized, parameterized_class
import tensorflow as tf
import numpy as np
import tempfile
import scipy.stats
from amber.utils import testing_utils
from amber import architect

# ----------
# architect.buffer
# ----------


class TestBuffer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBuffer, self).__init__(*args, **kwargs)
        self.buffer_getter = architect.buffer.get_buffer('ordinal')

    def setUp(self):
        super(TestBuffer, self).setUp()
        self.tempdir = tempfile.TemporaryDirectory()
        self.buffer = self.buffer_getter(max_size=4, is_squeeze_dim=True)
        self.state_space, _ = testing_utils.get_example_conv1d_space(num_layers=2)
        self.num_ops_per_layer = len(self.state_space[0])
        for _ in range(8):
            state, proba, act = self._get_data()
            self.buffer.store(state=state, prob=proba, action=act, reward=1)

    def tearDown(self):
        self.tempdir.cleanup()
        super(TestBuffer, self).tearDown()

    def _get_data(self):
        state = np.random.sample(4).reshape((1, 1, 4))
        proba = [
            # 1st layer operations, (1, n)
            np.random.sample(self.num_ops_per_layer).reshape((1, self.num_ops_per_layer)),
            # 2nd layer operations, (1, n)
            np.random.sample(self.num_ops_per_layer).reshape((1, self.num_ops_per_layer)),
            # 2nd layer residual con., (1, 1, 2)
            np.random.sample(2).reshape((1, 1, 2))
        ]
        act = np.random.choice(2, 3).astype('int')
        return state, proba, act

    def test_finish_path(self):
        self.buffer.finish_path(
            state_space=self.state_space,
            global_ep=0,
            working_dir=self.tempdir.name
        )
        # after path finish, long-term should be filled
        self.assertNotEqual(len(self.buffer.lt_abuffer), 0)
        self.assertNotEqual(len(self.buffer.lt_pbuffer), 0)
        self.assertNotEqual(len(self.buffer.lt_adbuffer), 0)

    def test_get(self):
        self.buffer.finish_path(
            state_space=self.state_space,
            global_ep=0,
            working_dir=self.tempdir.name
        )
        cnt = 0
        for data in self.buffer.get_data(bs=2):
            cnt += 1
            _, probas, acts, ads, rewards = data
            self.assertEqual(len(ads), 2)
            for pr in probas:
                self.assertEqual(len(pr), 2)
            self.assertEqual(len(acts), 2)
            self.assertEqual(len(rewards), 2)
            self.assertEqual(type(probas), list)
        self.assertEqual(cnt, 4)


class TestReplayBuffer(TestBuffer):
    def __init__(self, *args, **kwargs):
        super(TestReplayBuffer, self).__init__(*args, **kwargs)
        self.buffer_getter = architect.buffer.get_buffer('replay')


class TestMultiManagerBuffer(TestBuffer):
    def __init__(self, *args, **kwargs):
        super(TestMultiManagerBuffer, self).__init__(*args, **kwargs)
        self.buffer_getter = architect.buffer.get_buffer('multimanager')

    def setUp(self):
        super(TestBuffer, self).setUp()
        self.tempdir = tempfile.TemporaryDirectory()
        self.buffer = self.buffer_getter(max_size=4, is_squeeze_dim=True)
        self.state_space, _ = testing_utils.get_example_conv1d_space(num_layers=2)
        self.num_ops_per_layer = len(self.state_space[0])
        for manager_index in range(4):
            for i in range(8):
                _, proba, act = self._get_data()
                self.buffer.store(prob=proba, action=act, reward=manager_index,
                                  manager_index=manager_index,
                                  description=np.array([manager_index, manager_index**2]).reshape((1, 2))
                                  )

    def test_get(self):
        self.buffer.finish_path(
            state_space=self.state_space,
            global_ep=0,
            working_dir=self.tempdir.name
        )
        cnt = 0
        for data in self.buffer.get_data(bs=2):
            cnt += 1
            self.assertEqual(type(data), dict)
            probas = data["prob"]
            acts = data["action"]
            ads = data["advantage"]
            rewards = data["reward"]
            desc = data["description"]
            # make sure descriptors and rewards are aligned
            self.assertEqual(desc[0][0], rewards[0][0])
            self.assertEqual(desc[1][0], rewards[1][0])
            self.assertEqual(desc[0][1], rewards[0][0]**2)
            self.assertEqual(desc[1][1], rewards[1][0]**2)
            self.assertEqual(len(ads), 2)
            self.assertEqual(type(probas), list)
            for pr in probas:
                self.assertEqual(len(pr), 2)
            self.assertEqual(len(acts), 2)
            self.assertEqual(len(rewards), 2)
            self.assertEqual(len(desc), 2)
        self.assertEqual(cnt, 16)


# ----------
# architect.reward
# ----------


class TestReward(unittest.TestCase):
    # provide tuple data
    data = (
            None,
            np.array([0, 0, 0, 1, 1, 1])
        )

    # provide pseudo-model
    model = testing_utils.PseudoModel(
            pred_retr=np.array([-1, -1, -1, 1, 1, 1]),
            eval_retr={'val_loss': 0.5}
        )

    # provide pseudo-knowledge func
    knowledge_fn = testing_utils.PseudoKnowledge(k_val=0.2)


class TestAucReward(TestReward):
    def __init__(self, *args, **kwargs):
        super(TestAucReward, self).__init__(*args, **kwargs)
        self.reward_getter = architect.reward.LossAucReward

    @parameterized.expand([
        ('auc', 1),
        ('aupr', 1),
        (lambda y_true, y_score: scipy.stats.spearmanr(y_true, y_score)[0], 1),
        (lambda y_true, y_score: scipy.stats.pearsonr(y_true, y_score)[0], 1),
    ])
    def test_methods_call(self, method, expect_reward):
        reward_fn = self.reward_getter(method=method)
        reward, loss_and_metrics, reward_metrics = reward_fn(model=self.model, data=self.data)
        self.assertEqual(reward, expect_reward)
        self.assertTrue(hasattr(reward_fn, 'knowledge_function'))


class TestLossReward(TestReward):
    def __init__(self, *args, **kwargs):
        super(TestLossReward, self).__init__(*args, **kwargs)
        self.reward_getter = architect.reward.LossReward

    def test_methods_call(self):
        reward_fn = self.reward_getter()
        reward, loss_and_metrics, reward_metrics = reward_fn(model=self.model, data=self.data)
        self.assertEqual(reward, -0.5)
        self.assertTrue(hasattr(reward_fn, 'knowledge_function'))


@parameterized_class(('eval_retr', 'k_val', 'Lambda', 'exp_reward'), [
    (0.5, 0.2, 1, -0.7),
    (0.5, 0.2, 2, -0.9),
    (0.5, 0.2, 0, -0.5)
])
class TestKnowledgeReward(TestReward):

    def __init__(self, *args, **kwargs):
        super(TestKnowledgeReward, self).__init__(*args, **kwargs)
        self.reward_getter = architect.reward.KnowledgeReward
        self.model.eval_retr = self.eval_retr
        self.knowledge_fn.k_val = self.k_val

    def test_methods_call(self):
        reward_fn = self.reward_getter(knowledge_function=self.knowledge_fn, Lambda=self.Lambda)
        reward, loss_and_metrics, reward_metrics = reward_fn(model=self.model, data=self.data)
        self.assertEqual(reward, self.exp_reward)
        self.assertTrue(hasattr(reward_fn, 'knowledge_function'))


# ----------
# architect.manager
# ----------


class TestManager(testing_utils.TestCase):
    x = np.random.sample(10*4*1000).reshape((1000, 10, 4))
    y = np.random.sample(1000)

    def __init__(self, *args, **kwargs):
        super(TestManager, self).__init__(*args, **kwargs)
        self.reward_fn = testing_utils.PseudoReward()
        self.model_fn = testing_utils.PseudoConv1dModelBuilder(input_shape=(10, 4), output_units=1)
        self.store_fn = testing_utils.PseudoCaller()

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    @parameterized.expand([
        (1, architect.GeneralManager),
        (-1, architect.GeneralManager),
        (1, architect.EnasManager),
        (-1, architect.EnasManager)
    ])
    def test_get_reward(self, exp_reward, manager_getter):
        self.reward_fn.retr_val = exp_reward
        self.manager = manager_getter(
            train_data=(self.x, self.y),
            validation_data=(self.x, self.y),
            model_fn=self.model_fn,
            reward_fn=self.reward_fn,
            store_fn=self.store_fn,
            working_dir=self.tempdir.name,
            epochs=1,
            verbose=0
        )
        reward, loss_and_metrics = self.manager.get_rewards(trial=0, model_arc=[0, 0, 0])
        self.assertEqual(reward, exp_reward)

    def tearDown(self):
        self.tempdir.cleanup()
        super(TestManager, self).tearDown()


# ----------
# architect.store
# ----------


class TestStore(testing_utils.TestCase):
    x = np.random.sample(10*4*1000).reshape((1000, 10, 4))
    y = np.random.sample(1000)

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.trial = 0
        self.model = testing_utils.PseudoConv1dModelBuilder(input_shape=(10, 4), output_units=1)()
        model_checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.tempdir.name, 'temp_network.h5'),
            monitor='loss',
            save_best_only=True
        )
        self.history = self.model.fit(
            self.x, self.y,
            batch_size=100,
            epochs=5,
            verbose=0,
            callbacks=[model_checkpointer]
        )
        self.pred = self.model.predict(self.x, verbose=0)
        eval_retr = self.model.evaluate(self.x, self.y, verbose=0)
        self.loss_and_metrics = {'loss': eval_retr}

    @parameterized.expand([
        ('general', ('weights/trial_0/bestmodel.h5', 'weights/trial_0/pred.txt')),
        ('minimal', ('weights/trial_0/bestmodel.h5',)),
        ('model_plot', ('weights/trial_0/bestmodel.h5', 'weights/trial_0/pred.txt', 'weights/trial_0/model_arc.png'))
    ])
    def test_store_fn(self, store_name, files):
        store_fn = architect.store.get_store_fn(store_name)
        store_fn(
            trial=self.trial,
            model=self.model,
            hist=self.history,
            data=(self.x, self.y),
            pred=self.pred,
            loss_and_metrics=self.loss_and_metrics,
            working_dir=self.tempdir.name
        )
        for f in files:
            self.assertTrue(os.path.isfile(os.path.join(self.tempdir.name, f)))

    def tearDown(self):
        super(TestStore, self).tearDown()
        self.tempdir.cleanup()


if __name__ == '__main__':
    unittest.main()
