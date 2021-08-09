# -*- coding: UTF-8 -*-

"""
A pure Keras-implementation of NAS
ZZJ
Aug. 7, 2018
"""

import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.layers import Input, Lambda, Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Model

from .buffer import Buffer
from .generalController import BaseController
from .commonOps import get_kl_divergence_n_entropy, proximal_policy_optimization_loss


def get_optimizer(name: str, lr: float, decay: float = 0.995):
    if name.lower() == 'adam':
        return optimizers.Adam(lr, decay=decay)
    elif name.lower() == 'adagrad':
        return optimizers.Adagrad(lr, decay=decay)
    elif name.lower() == 'rmsprop':
        return optimizers.RMSprop(lr, decay=decay)
    elif name.lower() == 'sgd':
        return optimizers.SGD(lr, decay=decay)
    else:
        raise ValueError('Optimizer not supported')


def one_hot_encoder(val, num_classes, offset=0):
    val = np.array(val, dtype=np.int32)
    val -= offset
    if not val.shape:
        val = np.expand_dims(val, axis=0)
    assert all(val >= 0) and all(val < num_classes)
    tmp = np.zeros((val.shape[0], num_classes))
    for i, j in enumerate(val):
        tmp[i, j] = 1
    return tmp


def parse_action_str(action_onehot, state_space):
    return [state_space[i][int(j)] for i in range(len(state_space)) for j in range(len(action_onehot[i][0])) if
            action_onehot[i][0][int(j)] == 1]


def get_indexer(t):
    return Lambda(lambda x, t: x[:, t, :], arguments={'t': t}, output_shape=lambda s: (s[0], s[2]))


def build_actor(inputs, rnn_units, input_dim, maxlen, state_space, scope, trainable=True):
    rnn = LSTM(rnn_units, return_state=True, stateful=False, name=scope + '/NAScell', trainable=trainable)

    outputs = []  # pi
    action = []
    embeds = []

    state = None
    expand_layer = Lambda(lambda x: K.expand_dims(x, 1), output_shape=lambda s: (s[0], 1, s[1]),
                          name=scope + '/expand_layer')

    for t in range(maxlen):
        if t == 0:
            input_t = inputs
            input_t = Dense(input_dim, activation='linear', use_bias=False, name=scope + '/embed_input',
                            trainable=trainable)(input_t)
        else:
            input_t = embeds[-1]
            input_t = expand_layer(input_t)
        num_choices = len(state_space[t])
        output_t, h, c = rnn(input_t, initial_state=state)  # K.shape(inputs[0])[0])
        state = h, c
        logits = Dense(num_choices, name=scope + '/action_%i' % t, trainable=trainable)(output_t)
        output_t_prob = Activation('softmax', name=scope + '/sofmax_%i' % t)(logits)
        a = Lambda(lambda L: K.squeeze(tf.multinomial(L, 1), axis=1))(logits)
        a = Lambda(lambda L: K.one_hot(K.cast(L, 'int32'), num_classes=num_choices))(a)

        action.append(a)
        embed_t = Dense(input_dim, activation='linear', name=scope + '/embed_%i' % t, trainable=trainable)(
            output_t_prob)
        # embed_t = Dense(input_dim, activation='linear', name=scope+'/embed_%i' % t, trainable=trainable)(a)
        outputs.append(output_t_prob)
        embeds.append(embed_t)

    return outputs, action


class OperationController(BaseController):
    """
    example state_space for a 2-layer conv-net:
        state_space = [['conv3', 'conv5', 'conv7'], ['maxp2', 'avgp2'],
            ['conv3', 'conv5', 'conv7'], ['maxp2', 'avgp2']]
    """

    def __init__(self,
                 state_space,
                 controller_units,
                 embedding_dim=5,
                 optimizer='rmsprop',
                 discount_factor=0.0,
                 clip_val=0.2,        # for PPO clipping
                 beta=0.001,          # for entropy regularization
                 kl_threshold=0.05,   # for early stopping
                 train_pi_iter=100,   # num of substeps for training
                 lr_pi=0.005,
                 buffer_size=50,
                 batch_size=5,
                 verbose=0
                 ):
        self.state_space = state_space
        self.controller_units = controller_units
        self.embedding_dim = embedding_dim
        self.optimizer = get_optimizer(optimizer, lr_pi, 0.999)
        self.clip_val = clip_val
        self.beta = beta
        self.kl_threshold = kl_threshold
        self.global_controller_step = 0

        self.buffer = Buffer(buffer_size, discount_factor)

        self.kl_div = 0
        self.train_pi_iter = train_pi_iter
        self.batch_size = batch_size
        self.verbose = verbose

        self._build_sampler()
        self._build_trainer()

    def _build_sampler(self):
        maxlen = len(self.state_space)
        input_dim = self.embedding_dim
        last_output_dim = len(self.state_space.state_space[maxlen - 1])

        self.state_inputs = Input((1, last_output_dim), batch_shape=(None, 1, last_output_dim))  # states
        # build sampler model
        self.probs, self.action = build_actor(self.state_inputs, self.controller_units, input_dim, maxlen,
                                              self.state_space,
                                              scope='actor', trainable=True)
        self.model = Model(inputs=self.state_inputs, outputs=self.probs + self.action)

    def _build_trainer(self):
        # placeholders
        maxlen = len(self.probs)
        old_onehot_placeholder = [K.placeholder(shape=K.int_shape(self.probs[t]),
                                                name="old_onehot_%i" % t) for t in range(maxlen)]
        old_pred_placeholder = [K.placeholder(shape=K.int_shape(self.probs[t]),
                                              name="old_pred_%i" % t) for t in range(maxlen)]
        reward_placeholder = K.placeholder(shape=(None, 1),
                                           name="normal_reward")
        advantage_placeholder = K.placeholder((None, 1), name='advantage')

        loss = proximal_policy_optimization_loss(
            curr_prediction=self.probs,
            curr_onehot=self.action,
            old_prediction=old_pred_placeholder,
            old_onehotpred=old_onehot_placeholder,
            rewards=reward_placeholder,
            advantage=advantage_placeholder,
            clip_val=self.clip_val,
            beta=self.beta
        )

        kl_div, entropy = get_kl_divergence_n_entropy(old_pred_placeholder, self.probs, old_onehot_placeholder,
                                                      self.action)
        self.kl_div_fn = K.function(inputs=[self.state_inputs] + old_pred_placeholder + old_onehot_placeholder,
                                    outputs=[kl_div, entropy])

        updates = self.optimizer.get_updates(
            params=[w for w in self.model.trainable_weights if w.name.startswith('actor')],
            loss=loss)

        self.train_fn = K.function(inputs=[self.state_inputs, reward_placeholder, advantage_placeholder] +
                                          old_pred_placeholder + old_onehot_placeholder,
                                   outputs=[loss],
                                   updates=updates)

    def get_action(self, seed):
        maxlen = len(self.state_space)
        pred = self.model.predict(seed)
        prob = pred[:maxlen]
        onehot_action = pred[-maxlen:]

        return tuple(onehot_action), prob

    def store(self, state, prob, action, reward):

        self.buffer.store(state, prob, action, reward)

    def train(self, episode, working_dir):
        """
        called only when path finishes
        """

        self.buffer.finish_path(self.state_space, episode, working_dir)

        aloss = 0
        g_t = 0

        for epoch in range(self.train_pi_iter):

            t = 0
            kl_sum = 0
            ent_sum = 0

            # get data from buffer
            for s_batch, p_batch, a_batch, ad_batch, nr_batch in self.buffer.get_data(self.batch_size):

                feeds = [s_batch, nr_batch, ad_batch] + p_batch + a_batch
                aloss += self.train_fn(feeds)[0]

                curr_kl, curr_ent = self.kl_div_fn([s_batch] + p_batch + a_batch)
                kl_sum += curr_kl
                ent_sum += curr_ent

                t += 1
                g_t += 1

                if kl_sum / t > self.kl_threshold:
                    if self.verbose: print("     Early stopping at step {} as KL(old || new) = ".format(g_t), kl_sum / t)
                    return aloss / g_t

            if epoch % (self.train_pi_iter // 5) == 0:
                if self.verbose: print('     Epoch: {} Actor Loss: {} KL(old || new): {} Entropy(new) = {}'.format(epoch, aloss / g_t,
                                                                                                  kl_sum / t,
                                                                                                  ent_sum / t))

        return aloss / g_t  # + closs / self.train_v_iter

    def remove_files(self, files, working_dir='.'):
        for file in files:
            file = os.path.join(working_dir, file)
            if os.path.exists(file):
                os.remove(file)
