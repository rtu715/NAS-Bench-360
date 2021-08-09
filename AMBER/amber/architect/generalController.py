"""
General controller for searching computational operation per layer, and residual connection
"""

# Author       : ZZJ
# Last Update  : Aug. 16, 2020

import os
import sys

import h5py
import tensorflow as tf
if tf.__version__.startswith("2"):
    tf.compat.v1.disable_eager_execution()
    import tensorflow.compat.v1 as tf

from .buffer import get_buffer
from .commonOps import get_keras_train_ops
from .commonOps import get_kl_divergence_n_entropy
from .commonOps import proximal_policy_optimization_loss
from .commonOps import stack_lstm


class BaseController(object):
    """abstract class for controllers
    """

    def __init__(self, *args, **kwargs):
        # Abstract
        pass

    def __str__(self):
        return "AMBER Controller for architecture searching"

    def _create_weight(self, *args, **kwargs):
        raise NotImplementedError("Abstract method.")

    def _build_sampler(self, *args, **kwargs):
        raise NotImplementedError("Abstract method.")

    def _build_trainer(self, *args, **kwargs):
        raise NotImplementedError("Abstract method.")

    def store(self, *args, **kwargs):
        raise NotImplementedError("Abstract method.")

    def train(self, *args, **kwargs):
        raise NotImplementedError("Abstract method.")

    def remove_files(self, *args, **kwargs):
        raise NotImplementedError("Abstract method.")


class GeneralController(BaseController):
    """
    GeneralController for neural architecture search

    This class searches for two steps:
        - computational operations for each layer
        - skip connections for each layer from all previous layers [optional]

    It is a modified version of enas: https://github.com/melodyguan/enas . Notable modifications include: dissection of
    sampling and training processes to enable better understanding of controller behaviors, buffering and logging;
    loss function can be optimized by either REINFORCE or PPO.

    TODO
    ----------
    Refactor the rest of the attributes to private.


    Parameters
    ----------
    model_space : amber.architect.ModelSpace
        A ModelSpace object constructed to perform architecture search for.

    with_skip_connection : bool
        If false, will not search residual connections and only search for computation operations per layer. Default is
        True.

    share_embedding : dict
        a Dictionary defining which child-net layers will share the softmax and embedding weights during Controller
        training and sampling. For example, ``{1:0, 2:0}`` means layer 1 and 2 will share the embedding with layer 0.

    use_ppo_loss : bool
        If true, use PPO loss for optimization instead of REINFORCE. Default is False.

    kl_threshold : float
        If KL-divergence between the sampling probabilities of updated controller parameters and that of original
        parameters exceeds kl_threshold within a single controller training step, triggers early-stopping to halt the
        controller training. Default is 0.05.

    buffer_size : int
        amber.architect.Buffer stores only the sampled architectures from the last ``buffer_size`` number of from previous
        controller steps, where each step has a number of sampled architectures as specified in ``amber.architect.ControllerTrainEnv``.

    batch_size : int
        How many architectures in a batch to train the controller

    session : tf.Session
        The session where the controller tensors is placed

    train_pi_iter : int
        The number of epochs/iterations to train controller policy in one controller step.

    lstm_size : int
        The size of hidden units for stacked LSTM, i.e. controller RNN.

    lstm_num_layers : int
        The number of stacked layers for stacked LSTM, i.e. controller RNN.

    lstm_keep_prob : float
        keep_prob = 1 - dropout probability for stacked LSTM.

    tanh_constant : float
        If not None, the logits for each multivariate classification will be transformed by ``tf.tanh`` then multiplied by
        tanh_constant. This can avoid over-confident controllers asserting probability=1 or 0 caused by logit going to +/- inf.
        Default is None.

    temperature : float
        The temperature is a scale factor to logits. Higher temperature will flatten the probabilities among different
        classes, while lower temperature will freeze them. Default is None, i.e. 1.

    optim_algo : str
        Optimizer for controller RNN. Can choose from ["adam", "sgd", "rmsprop"]. Default is "adam".

    skip_target : float
        The expected proportion of skip connections, i.e. the proportion of 1's in the skip/extra
        connections in the output `arc_seq`

    skip_weight : float
        The weight for skip connection kl-divergence from the expected `skip_target`

    name : str
        The name for this Controller instance; all ``tf.Tensors`` will be placed under this VariableScope. This name
        determines which tensors will be initialized when a new Controller instance is created.


    Attributes
    ----------
    weights : list of tf.Variable
        The list of all trainable ``tf.Variable`` in this controller

    model_space : amber.architect.ModelSpace
        The model space which the controller will be searching from.

    buffer : amber.architect.Buffer
        The Buffer object stores the history architectures, computes the rewards, and gets feed dict for training.

    session : tf.Session
        The reference to the session that hosts this controller instance.



    """

    def __init__(self, model_space, buffer_type='ordinal', with_skip_connection=True, share_embedding=None,
                 use_ppo_loss=False, kl_threshold=0.05, skip_connection_unique_connection=False, buffer_size=15,
                 batch_size=5, session=None, train_pi_iter=20, lstm_size=32, lstm_num_layers=2, lstm_keep_prob=1.0,
                 tanh_constant=None, temperature=None, optim_algo="adam", skip_target=0.8, skip_weight=None,
                 rescale_advantage_by_reward=False, name="controller", verbose=0, **kwargs):
        super().__init__(**kwargs)

        self.model_space = model_space
        # -----
        # FOR LEGACY ATTRIBUTES
        self.state_space = model_space
        # -----
        self.share_embedding = share_embedding
        self.with_skip_connection = with_skip_connection
        self.num_layers = len(model_space)
        self.num_choices_per_layer = [len(model_space[i]) for i in range(self.num_layers)]
        self.skip_connection_unique_connection = skip_connection_unique_connection

        buffer_fn = get_buffer(buffer_type)
        self.buffer = buffer_fn(max_size=buffer_size,
                                #ewa_beta=max(1 - 1. / buffer_size, 0.9),
                                discount_factor=0.,
                                is_squeeze_dim=True,
                                rescale_advantage_by_reward=rescale_advantage_by_reward)
        self.batch_size = batch_size
        self.verbose = verbose

        # need to use the same session throughout one App; ZZ 2020.3.2
        assert session is not None
        self.session = session if session else tf.Session()
        self.train_pi_iter = train_pi_iter
        self.use_ppo_loss = use_ppo_loss
        self.kl_threshold = kl_threshold

        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_keep_prob = lstm_keep_prob
        self.tanh_constant = tanh_constant
        self.temperature = temperature

        self.skip_target = skip_target
        self.skip_weight = skip_weight
        if self.skip_weight is not None:
            assert self.with_skip_connection, "If skip_weight is not None, must have with_skip_connection=True"

        self.optim_algo = optim_algo
        self.name = name
        self.loss = 0

        with tf.device("/cpu:0"):
            with tf.variable_scope(self.name):
                self._create_weight()
                self._build_sampler()
                self._build_trainer()
                self._build_train_op()
        # initialize variables in this scope
        self.weights = [var for var in tf.trainable_variables() if var.name.startswith(self.name)]
        self.session.run(tf.variables_initializer(self.weights))

    def __str__(self):
        s = "GeneralController '%s' for %s" % (self.name, self.model_space)
        return s

    def _create_weight(self):
        """Private method for creating tensors; called at initialization"""
        initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope("create_weights", initializer=initializer):
            with tf.variable_scope("lstm", reuse=False):
                self.w_lstm = []
                for layer_id in range(self.lstm_num_layers):
                    with tf.variable_scope("lstm_layer_{}".format(layer_id)):
                        w = tf.get_variable(
                            "w", [2 * self.lstm_size, 4 * self.lstm_size])
                        self.w_lstm.append(w)

            # g_emb: initial controller hidden state tensor; to be learned
            self.g_emb = tf.get_variable("g_emb", [1, self.lstm_size])

            # w_emb: embedding for computational operations
            self.w_emb = {"start": []}

            with tf.variable_scope("emb"):
                for layer_id in range(self.num_layers):
                    with tf.variable_scope("layer_{}".format(layer_id)):
                        if self.share_embedding:
                            if layer_id not in self.share_embedding:
                                self.w_emb["start"].append(tf.get_variable(
                                    "w_start", [self.num_choices_per_layer[layer_id], self.lstm_size]))
                            else:
                                shared_id = self.share_embedding[layer_id]
                                assert shared_id < layer_id, \
                                    "You turned on `share_embedding`, but specified the layer %i " \
                                    "to be shared with layer %i, which is not built yet" % (layer_id, shared_id)
                                self.w_emb["start"].append(self.w_emb["start"][shared_id])

                        else:
                            self.w_emb["start"].append(tf.get_variable(
                                "w_start", [self.num_choices_per_layer[layer_id], self.lstm_size]))

            # w_soft: dictionary of tensors for transforming RNN hiddenstates to softmax classifier
            self.w_soft = {"start": []}
            with tf.variable_scope("softmax"):
                for layer_id in range(self.num_layers):
                    if self.share_embedding:
                        if layer_id not in self.share_embedding:
                            with tf.variable_scope("layer_{}".format(layer_id)):
                                self.w_soft["start"].append(tf.get_variable(
                                    "w_start", [self.lstm_size, self.num_choices_per_layer[layer_id]]))
                        else:
                            shared_id = self.share_embedding[layer_id]
                            assert shared_id < layer_id, \
                                "You turned on `share_embedding`, but specified the layer %i " \
                                "to be shared with layer %i, which is not built yet" % (layer_id, shared_id)
                            self.w_soft["start"].append(self.w_soft['start'][shared_id])
                    else:
                        with tf.variable_scope("layer_{}".format(layer_id)):
                            self.w_soft["start"].append(tf.get_variable(
                                "w_start", [self.lstm_size, self.num_choices_per_layer[layer_id]]))

            #  w_attn_1/2, v_attn: for sampling skip connections
            if self.with_skip_connection:
                with tf.variable_scope("attention"):
                    self.w_attn_1 = tf.get_variable("w_1", [self.lstm_size, self.lstm_size])
                    self.w_attn_2 = tf.get_variable("w_2", [self.lstm_size, self.lstm_size])
                    self.v_attn = tf.get_variable("v", [self.lstm_size, 1])
            else:
                self.w_attn_1 = None
                self.w_attn_2 = None
                self.v_attn = None

    def _build_sampler(self):
        """Build the sampler ops and the log_prob ops.

        For sampler, the architecture sequence is randomly sampled, and only sample one architecture at each call to
        fill in self.sample_arc
        """
        anchors = []
        anchors_w_1 = []

        arc_seq = []
        hidden_states = []
        entropys = []
        probs_ = []
        log_probs = []
        skip_count = []
        skip_penaltys = []

        prev_c = [tf.zeros([1, self.lstm_size], tf.float32) for _ in
                  range(self.lstm_num_layers)]
        prev_h = [tf.zeros([1, self.lstm_size], tf.float32) for _ in
                  range(self.lstm_num_layers)]

        inputs = self.g_emb
        skip_targets = tf.constant([1.0 - self.skip_target, self.skip_target],
                                   dtype=tf.float32)
        skip_conn_record = []

        for layer_id in range(self.num_layers):
            # STEP 1: for each layer, sample operations first
            next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            prev_c, prev_h = next_c, next_h
            hidden_states.append(prev_h)

            logit = tf.matmul(next_h[-1], self.w_soft["start"][layer_id])  # out_filter x 1
            if self.temperature is not None:
                logit /= self.temperature
            if self.tanh_constant is not None:
                logit = self.tanh_constant * tf.tanh(logit)
            probs_.append(tf.nn.softmax(logit))
            start = tf.multinomial(logit, 1)
            start = tf.to_int32(start)
            start = tf.reshape(start, [1])
            arc_seq.append(start)
            log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logit, labels=start)
            log_probs.append(log_prob)
            entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
            entropys.append(entropy)
            # inputs: get a row slice of [out_filter[i], lstm_size]
            inputs = tf.nn.embedding_lookup(self.w_emb["start"][layer_id], start)
            # END STEP 1

            # STEP 2: sample the connections, unless the first layer
            # the number `skip` of each layer grows as layer_id grows
            if self.with_skip_connection:
                if layer_id > 0:
                    next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                    prev_c, prev_h = next_c, next_h
                    hidden_states.append(prev_h)

                    query = tf.concat(anchors_w_1, axis=0)  # layer_id x lstm_size
                    # w_attn_2: lstm_size x lstm_size
                    query = tf.tanh(query + tf.matmul(next_h[-1], self.w_attn_2))  # query: layer_id x lstm_size
                    # P(Layer j is an input to layer i) = sigmoid(v^T %*% tanh(W_prev ∗ h_j + W_curr ∗ h_i))
                    query = tf.matmul(query, self.v_attn)  # query: layer_id x 1
                    if self.skip_connection_unique_connection:
                        mask = tf.stop_gradient(tf.reduce_sum(tf.stack(skip_conn_record), axis=0))
                        mask = tf.slice(mask, begin=[0], size=[layer_id])
                        mask1 = tf.greater(mask, 0)
                        query = tf.where(mask1, y=query, x=tf.fill(tf.shape(query), -10000.))
                    logit = tf.concat([-query, query], axis=1)  # logit: layer_id x 2
                    if self.temperature is not None:
                        logit /= self.temperature
                    if self.tanh_constant is not None:
                        logit = self.tanh_constant * tf.tanh(logit)

                    probs_.append(tf.expand_dims(tf.nn.softmax(logit), axis=0))
                    skip = tf.multinomial(logit, 1)  # layer_id x 1 of booleans
                    skip = tf.to_int32(skip)
                    skip = tf.reshape(skip, [layer_id])
                    arc_seq.append(skip)
                    skip_conn_record.append(
                        tf.concat([tf.cast(skip, tf.float32), tf.zeros(self.num_layers - layer_id)], axis=0))

                    skip_prob = tf.sigmoid(logit)
                    kl = skip_prob * tf.log(skip_prob / skip_targets)
                    kl = tf.reduce_sum(kl)
                    skip_penaltys.append(kl)

                    log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logit, labels=skip)
                    log_probs.append(tf.reshape(tf.reduce_sum(log_prob), [-1]))

                    entropy = tf.stop_gradient(
                        tf.reshape(tf.reduce_sum(log_prob * tf.exp(-log_prob)), [-1]))
                    entropys.append(entropy)

                    skip = tf.to_float(skip)
                    skip = tf.reshape(skip, [1, layer_id])
                    skip_count.append(tf.reduce_sum(skip))
                    inputs = tf.matmul(skip, tf.concat(anchors, axis=0))
                    inputs /= (1.0 + tf.reduce_sum(skip))
                else:
                    skip_conn_record.append(tf.zeros(self.num_layers, 1))

                anchors.append(next_h[-1])
                # next_h: 1 x lstm_size
                # anchors_w_1: 1 x lstm_size
                anchors_w_1.append(tf.matmul(next_h[-1], self.w_attn_1))
                # added Sep.28.2019; removed as inputs for next layer, Nov.21.2019
                # next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                # prev_c, prev_h = next_c, next_h
                # hidden_states.append(prev_h)

            # END STEP 2

        # for DEBUG use
        self.anchors = anchors
        self.anchors_w_1 = anchors_w_1
        self.sample_hidden_states = hidden_states

        # for class attr.
        arc_seq = tf.concat(arc_seq, axis=0)
        self.sample_arc = tf.reshape(arc_seq, [-1])
        entropys = tf.stack(entropys)
        self.sample_entropy = tf.reduce_sum(entropys)
        log_probs = tf.stack(log_probs)
        self.sample_log_prob = tf.reduce_sum(log_probs)
        skip_count = tf.stack(skip_count)
        self.skip_count = tf.reduce_sum(skip_count)
        skip_penaltys = tf.stack(skip_penaltys)
        self.skip_penaltys = tf.reduce_mean(skip_penaltys)
        self.sample_probs = probs_

    def _build_trainer(self):
        """"Build the trainer ops and the log_prob ops.

        For trainer, the input architectures are ``tf.placeholder`` to receive previous architectures from buffer.
        It also supports batch computation.
        """
        anchors = []
        anchors_w_1 = []
        probs_ = []

        ops_each_layer = 1
        total_arc_len = sum(
            [ops_each_layer] +   # first layer
            [ops_each_layer + i * self.with_skip_connection for i in range(1, self.num_layers)]  # rest layers
        )
        self.total_arc_len = total_arc_len
        self.input_arc = [tf.placeholder(shape=(None, 1), dtype=tf.int32, name='arc_{}'.format(i))
                          for i in range(total_arc_len)]

        batch_size = tf.shape(self.input_arc[0])[0]
        entropys = []
        log_probs = []
        skip_count = []
        skip_penaltys = []

        prev_c = [tf.zeros([batch_size, self.lstm_size], tf.float32) for _ in
                  range(self.lstm_num_layers)]
        prev_h = [tf.zeros([batch_size, self.lstm_size], tf.float32) for _ in
                  range(self.lstm_num_layers)]
        # only expand `g_emb` if necessary
        g_emb_nrow = self.g_emb.shape[0] if type(self.g_emb.shape[0]) is int \
            else self.g_emb.shape[0].value
        if self.g_emb.shape[0] is not None and g_emb_nrow == 1:
            inputs = tf.matmul(tf.ones((batch_size, 1)), self.g_emb)
        else:
            inputs = self.g_emb
        skip_targets = tf.constant([1.0 - self.skip_target, self.skip_target],
                                   dtype=tf.float32)

        arc_pointer = 0
        skip_conn_record = []
        hidden_states = []
        for layer_id in range(self.num_layers):

            # STEP 1: compute log-prob for operations
            next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            prev_c, prev_h = next_c, next_h
            hidden_states.append(prev_h)

            logit = tf.matmul(next_h[-1], self.w_soft["start"][layer_id])  # batch_size x num_choices_layer_i
            if self.temperature is not None:
                logit /= self.temperature
            if self.tanh_constant is not None:
                logit = self.tanh_constant * tf.tanh(logit)
            start = self.input_arc[arc_pointer]
            start = tf.reshape(start, [batch_size])
            probs_.append(tf.nn.softmax(logit))

            log_prob1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logit, labels=start)
            log_probs.append(log_prob1)
            entropy = tf.stop_gradient(log_prob1 * tf.exp(-log_prob1))
            entropys.append(entropy)
            # inputs: get a row slice of [out_filter[i], lstm_size]
            # inputs = tf.nn.embedding_lookup(self.w_emb["start"][branch_id], start)
            inputs = tf.nn.embedding_lookup(self.w_emb["start"][layer_id], start)
            # END STEP 1

            # STEP 2: compute log-prob for skip connections, unless the first layer
            if self.with_skip_connection:
                if layer_id > 0:

                    next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                    prev_c, prev_h = next_c, next_h
                    hidden_states.append(prev_h)

                    query = tf.transpose(tf.stack(anchors_w_1), [1, 0, 2])  # batch_size x layer_id x lstm_size
                    # w_attn_2: lstm_size x lstm_size
                    # P(Layer j is an input to layer i) = sigmoid(v^T %*% tanh(W_prev ∗ h_j + W_curr ∗ h_i))
                    query = tf.tanh(
                        query + tf.expand_dims(tf.matmul(next_h[-1], self.w_attn_2),
                                               axis=1))  # query: layer_id x lstm_size
                    query = tf.reshape(query, (batch_size * layer_id, self.lstm_size))
                    query = tf.matmul(query, self.v_attn)  # query: batch_size*layer_id x 1

                    if self.skip_connection_unique_connection:
                        mask = tf.stop_gradient(tf.reduce_sum(tf.stack(skip_conn_record), axis=0))
                        mask = tf.slice(mask, begin=[0, 0], size=[batch_size, layer_id])
                        mask = tf.reshape(mask, (batch_size * layer_id, 1))
                        mask1 = tf.greater(mask, 0)
                        query = tf.where(mask1, y=query, x=tf.fill(tf.shape(query), -10000.))

                    logit = tf.concat([-query, query], axis=1)  # logit: batch_size*layer_id x 2
                    if self.temperature is not None:
                        logit /= self.temperature
                    if self.tanh_constant is not None:
                        logit = self.tanh_constant * tf.tanh(logit)

                    probs_.append(tf.reshape(tf.nn.softmax(logit), [batch_size, layer_id, 2]))

                    skip = self.input_arc[(arc_pointer + ops_each_layer): (arc_pointer + ops_each_layer + layer_id)]
                    # print(layer_id, (arc_pointer+2), (arc_pointer+2 + layer_id), skip)
                    skip = tf.reshape(tf.transpose(skip), [batch_size * layer_id])
                    skip = tf.to_int32(skip)

                    skip_prob = tf.sigmoid(logit)
                    kl = skip_prob * tf.log(skip_prob / skip_targets)  # (batch_size*layer_id, 2)
                    kl = tf.reduce_sum(kl, axis=1)    # (batch_size*layer_id,)
                    kl = tf.reshape(kl, [batch_size, -1])  # (batch_size, layer_id)
                    skip_penaltys.append(kl)

                    log_prob3 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logit, labels=skip)
                    log_prob3 = tf.reshape(log_prob3, [batch_size, -1])
                    log_probs.append(tf.reduce_sum(log_prob3, axis=1))

                    entropy = tf.stop_gradient(
                        tf.reduce_sum(log_prob3 * tf.exp(-log_prob3), axis=1))
                    entropys.append(entropy)

                    skip = tf.to_float(skip)
                    skip = tf.reshape(skip, [batch_size, 1, layer_id])
                    skip_count.append(tf.reduce_sum(skip, axis=2))

                    anchors_ = tf.stack(anchors)
                    anchors_ = tf.transpose(anchors_, [1, 0, 2])  # batch_size, layer_id, lstm_size
                    inputs = tf.matmul(skip, anchors_)  # batch_size, 1, lstm_size
                    inputs = tf.squeeze(inputs, axis=1)
                    inputs /= (1.0 + tf.reduce_sum(skip, axis=2))  # batch_size, lstm_size

                else:
                    skip_conn_record.append(tf.zeros((batch_size, self.num_layers)))

                # next_h: batch_size x lstm_size
                anchors.append(next_h[-1])
                # anchors_w_1: batch_size x lstm_size
                anchors_w_1.append(tf.matmul(next_h[-1], self.w_attn_1))
                # added 9.28.2019; removed 11.21.2019
                # next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                # prev_c, prev_h = next_c, next_h

            arc_pointer += ops_each_layer + layer_id * self.with_skip_connection
        # END STEP 2

        # for DEBUG use
        self.train_hidden_states = hidden_states

        # for class attributes
        self.entropys = tf.stack(entropys)
        self.onehot_probs = probs_
        log_probs = tf.stack(log_probs)
        self.onehot_log_prob = tf.reduce_sum(log_probs, axis=0)
        skip_count = tf.stack(skip_count)
        self.onehot_skip_count = tf.reduce_sum(skip_count, axis=0)
        skip_penaltys_flat = [tf.reduce_mean(x, axis=1) for x in skip_penaltys] # from (num_layer-1, batch_size, layer_id) to (num_layer-1, batch_size); layer_id makes each tensor of varying lengths in the list
        self.onehot_skip_penaltys = tf.reduce_mean(skip_penaltys_flat, axis=0)  # (batch_size,)

    def _build_train_op(self):
        """build train_op based on either REINFORCE or PPO
        """
        self.advantage = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="advantage")
        self.reward = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="reward")

        normalize = tf.to_float(self.num_layers * (self.num_layers - 1) / 2)
        self.skip_rate = tf.to_float(self.skip_count) / normalize

        self.input_arc_onehot = self.convert_arc_to_onehot(self)
        self.old_probs = [tf.placeholder(shape=self.onehot_probs[i].shape, dtype=tf.float32, name="old_prob_%i" % i) for
                          i in range(len(self.onehot_probs))]
        if self.skip_weight is not None:
            self.loss += self.skip_weight * tf.reduce_mean(self.onehot_skip_penaltys)
        if self.use_ppo_loss:
            self.loss += proximal_policy_optimization_loss(
                curr_prediction=self.onehot_probs,
                curr_onehot=self.input_arc_onehot,
                old_prediction=self.old_probs,
                old_onehotpred=self.input_arc_onehot,
                rewards=self.reward,
                advantage=self.advantage,
                clip_val=0.2)
        else:
            self.loss += tf.reshape(tf.tensordot(self.onehot_log_prob, self.advantage, axes=1), [])

        self.kl_div, self.ent = get_kl_divergence_n_entropy(curr_prediction=self.onehot_probs,
                                                            old_prediction=self.old_probs,
                                                            curr_onehot=self.input_arc_onehot,
                                                            old_onehotpred=self.input_arc_onehot)
        self.train_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="train_step")
        tf_variables = [var
                        for var in tf.trainable_variables() if var.name.startswith(self.name)]

        self.train_op, self.lr, self.grad_norm, self.optimizer = get_keras_train_ops(
            loss=self.loss,
            tf_variables=tf_variables,
            optim_algo=self.optim_algo
        )

    def get_action(self, *args, **kwargs):
        """Get a sampled architecture/action and its corresponding probabilities give current controller policy parameters.

        The generated architecture is the out-going information from controller to manager. which in turn will feedback
        the reward signal for storage and training by the controller.

        Parameters
        ----------
        None

        Returns
        ----------
        onehots : list
            The sampled architecture sequence. In particular, the architecture sequence is ordered as::

                [categorical_operation_0,
                categorical_operation_1, binary_skip_0,
                categorical_operation_2, binary_skip_0, binary_skip_1,
                ...]


        probs : list of ndarray
            The probabilities associated with each sampled operation and residual connection. Shapes will vary depending
            on each layer's specification in ModelSpace for operation, and the layer number for residual connections.
        """
        probs, onehots = self.session.run([self.sample_probs, self.sample_arc])
        return onehots, probs

    def train(self, episode, working_dir):
        """Train the controller policy parameters for one step.

        Parameters
        ----------
        episode : int
            Total number of epochs to train the controller. Each epoch will iterate over all architectures stored in buffer.

        working_dir : str
            Filepath to working directory to store (possible) intermediate results

        Returns
        -------
        aloss : float
            Average controller loss for this train step

        Notes
        -----
        Consider renaming this method to ``train_step()`` to better reflect its function, and avoid confusion with the
        training function in environment ``ControllerTrainEnv.train()``
        """
        try:
            self.buffer.finish_path(self.model_space, episode, working_dir)
        except Exception as e:
            print("cannot finish path in buffer because: %s" % e)
            sys.exit(1)
        aloss = 0
        g_t = 0

        for epoch in range(self.train_pi_iter):
            t = 0
            kl_sum = 0
            ent_sum = 0
            # get data from buffer
            for s_batch, p_batch, a_batch, ad_batch, nr_batch in self.buffer.get_data(self.batch_size):
                feed_dict = {self.input_arc[i]: a_batch[:, [i]]
                             for i in range(a_batch.shape[1])}
                feed_dict.update({self.advantage: ad_batch})
                feed_dict.update({self.old_probs[i]: p_batch[i]
                                  for i in range(len(self.old_probs))})
                feed_dict.update({self.reward: nr_batch})

                self.session.run(self.train_op, feed_dict=feed_dict)
                curr_loss, curr_kl, curr_ent = self.session.run([self.loss, self.kl_div, self.ent], feed_dict=feed_dict)
                aloss += curr_loss
                kl_sum += curr_kl
                ent_sum += curr_ent
                t += 1
                g_t += 1

                if kl_sum / t > self.kl_threshold and epoch > 0 and self.verbose > 0:
                    print("     Early stopping at step {} as KL(old || new) = ".format(g_t), kl_sum / t)
                    return aloss / g_t

            if epoch % max(1, (self.train_pi_iter // 5)) == 0 and self.verbose > 0:
                print("     Epoch: {} Actor Loss: {} KL(old || new): {} Entropy(new) = {}".format(
                    epoch, aloss / g_t,
                    kl_sum / t,
                    ent_sum / t)
                )

        return aloss / g_t

    def store(self, state, prob, action, reward, *args, **kwargs):
        """Store all necessary information and rewards for a given architecture

        This is the receiving method for controller to interact with manager by storing the rewards for a given architecture.
        The architecture and its probabilities can be generated by ``get_action()`` method.

        Parameters
        ----------
        state : list
            The state for which the action and probabilities are drawn.

        prob : list of ndarray
            A list of probabilities for each operation and skip connections.

        action : list
            A list of architecture tokens ordered as::

                [categorical_operation_0,
                categorical_operation_1, binary_skip_0,
                categorical_operation_2, binary_skip_0, binary_skip_1,
                ...]

        reward : float
            Reward for this architecture, as evaluated by ``amber.architect.manager``

        Returns
        -------
        None

        """
        self.buffer.store(state=state, prob=prob, action=action, reward=reward)
        return

    @staticmethod
    def remove_files(files, working_dir='.'):
        """Static method for removing files

        Parameters
        ----------
        files : list of str
            files to be removed

        working_dir : str
            filepath to working directory

        Returns
        -------
        None
        """
        for file in files:
            file = os.path.join(working_dir, file)
            if os.path.exists(file):
                os.remove(file)

    def save_weights(self, filepath, **kwargs):
        """Save current controller weights to a hdf5 file

        Parameters
        ----------
        filepath : str
            file path to save the weights

        Returns
        -------
        None
        """
        weights = self.get_weights()
        with h5py.File(filepath, "w") as hf:
            for i, d in enumerate(weights):
                hf.create_dataset(name=self.weights[i].name, data=d)

    def load_weights(self, filepath, **kwargs):
        """Load the controller weights from a hdf5 file

        Parameters
        ----------
        filepath : str
            file path to saved weights

        Returns
        -------
        None
        """
        weights = []
        with h5py.File(filepath, 'r') as hf:
            for i in range(len(self.weights)):
                key = self.weights[i].name
                weights.append(hf.get(key).value)
        self.set_weights(weights)

    def get_weights(self, **kwargs):
        """Get the current controller weights in a numpy array

        Parameters
        ----------
        None

        Returns
        -------
        weights : list
            A list of numpy array for each weights in controller
        """
        weights = self.session.run(self.weights)
        return weights

    def set_weights(self, weights, **kwargs):
        """Set the current controller weights

        Parameters
        ----------
        weights : list of numpy.ndarray
            A list of numpy array for each weights in controller

        Returns
        -------
        None
        """
        assign_ops = []
        for i in range(len(self.weights)):
            assign_ops.append(tf.assign(self.weights[i], weights[i]))
        self.session.run(assign_ops)

    @staticmethod
    def convert_arc_to_onehot(controller):
        """Convert a categorical architecture sequence to a one-hot encoded architecture sequence

        Parameters
        ----------
        controller : amber.architect.controller
            An instance of controller

        Returns
        -------
        onehot_list : list
            a one-hot encoded architecture sequence
        """
        with_skip_connection = controller.with_skip_connection
        if hasattr(controller, 'with_input_blocks'):
            with_input_blocks = controller.with_input_blocks
            num_input_blocks = controller.num_input_blocks
        else:
            with_input_blocks = False
            num_input_blocks = 1
        arc_seq = controller.input_arc
        model_space = controller.model_space
        onehot_list = []
        arc_pointer = 0
        for layer_id in range(len(model_space)):
            # print("layer_type ",arc_pointer)
            onehot_list.append(tf.squeeze(tf.one_hot(arc_seq[arc_pointer], depth=len(model_space[layer_id])), axis=1))
            if with_input_blocks:
                inp_blocks_idx = arc_pointer + 1, arc_pointer + 1 + num_input_blocks * with_input_blocks
                tmp = []
                for i in range(inp_blocks_idx[0], inp_blocks_idx[1]):
                    # print("input block ",i)
                    tmp.append(tf.squeeze(tf.one_hot(arc_seq[i], 2), axis=1))
                onehot_list.append(tf.transpose(tf.stack(tmp), [1, 0, 2]))
            if layer_id > 0 and with_skip_connection:
                skip_con_idx = arc_pointer + 1 + num_input_blocks * with_input_blocks, \
                               arc_pointer + 1 + num_input_blocks * with_input_blocks + layer_id * with_skip_connection
                tmp = []
                for i in range(skip_con_idx[0], skip_con_idx[1]):
                    # print("skip con ",i)
                    tmp.append(tf.squeeze(tf.one_hot(arc_seq[i], 2), axis=1))
                onehot_list.append(tf.transpose(tf.stack(tmp), [1, 0, 2]))
            arc_pointer += 1 + num_input_blocks * with_input_blocks + layer_id * with_skip_connection
        return onehot_list
