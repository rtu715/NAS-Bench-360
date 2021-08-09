import tensorflow as tf

from .generalController import GeneralController, stack_lstm


class MultiInputController(GeneralController):
    """
    DOCSTRING

    Parameters
    ----------
    model_space:
    with_skip_connection:
    with_input_blocks:
    share_embedding: dict
        a Dictionary defining which child-net layers will share the softmax and
        embedding weights during Controller training and sampling
    use_ppo_loss:
    kl_threshold:
    num_input_blocks:
    input_block_unique_connection:
    buffer_size:
    batch_size:
    session:
    train_pi_iter:
    lstm_size:
    lstm_num_layers:
    lstm_keep_prob:
    tanh_constant:
    temperature:
    lr_init:
    lr_dec_start:
    lr_dec_every:
    lr_dec_rate:
    l2_reg:
    clip_mode:
    grad_bound:
    use_critic:
    bl_dec:
    optim_algo:
    sync_replicas:
    num_aggregate:
    num_replicas:
    skip_target: float
        the expected proportion of skip connections, i.e. the proportion of 1's in the skip/extra
        connections in the output `arc_seq`
    skip_weight: float
        the weight for skip connection kl-divergence from the expected `skip_target`
    name: str
        name for the instance; also used for all tensor variable scopes

    Attributes
    ----------
    g_emb: tf.Tensor
        initial controller hidden state tensor; to be learned
    Placeholder

    Note
    ----------
    This class derived from `GeneralController` adds the input feature block selection upon the Base class. Since the
    selection is inherently embedded in the NAS cell rolling-out, the sampler and trainer methods are overwritten.

    TODO:
        needs to evaluate how different ways of connecting inputs will affect search performance; e.g. connect input
        before operation or after?
    """

    def __init__(self, model_space, buffer_type='ordinal', with_skip_connection=True, with_input_blocks=True,
                 share_embedding=None, use_ppo_loss=False, kl_threshold=0.05, num_input_blocks=2,
                 input_block_unique_connection=True, skip_connection_unique_connection=False, buffer_size=15,
                 batch_size=5, session=None, train_pi_iter=20, lstm_size=32, lstm_num_layers=2, lstm_keep_prob=1.0,
                 tanh_constant=None, temperature=None, skip_target=0.8, skip_weight=0.5, optim_algo="adam",
                 name="controller", *args, **kwargs):

        # unique attributes to MultiInputController:
        self.with_input_blocks = with_input_blocks
        self.num_input_blocks = num_input_blocks
        self.input_block_unique_connection = input_block_unique_connection

        super().__init__(model_space=model_space, buffer_type=buffer_type, with_skip_connection=with_skip_connection,
                         share_embedding=share_embedding, use_ppo_loss=use_ppo_loss, kl_threshold=kl_threshold,
                         skip_connection_unique_connection=skip_connection_unique_connection, buffer_size=buffer_size,
                         batch_size=batch_size, session=session, train_pi_iter=train_pi_iter, lstm_size=lstm_size,
                         lstm_num_layers=lstm_num_layers, lstm_keep_prob=lstm_keep_prob, tanh_constant=tanh_constant,
                         temperature=temperature, optim_algo=optim_algo, skip_target=skip_target,
                         skip_weight=skip_weight, name=name, **kwargs)

    def _create_weight(self):
        super()._create_weight()
        if self.with_input_blocks:
            with tf.variable_scope("input", initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)):
                # input_emb: embedding for input blocks, if present
                self.input_emb = tf.get_variable("inp_emb", [self.num_input_blocks, self.lstm_size])
                # w_soft['input']: transforming RNN hiddenstates to input softmax/binary selection
                self.w_soft["input"] = tf.get_variable("w_input", [self.lstm_size, self.num_input_blocks])

    # overwrite
    def _build_sampler(self):
        """Build the sampler ops and the log_prob ops."""
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
        input_block_record = []
        skip_conn_record = []

        for layer_id in range(self.num_layers):
            # BLOCK 1: for each layer, sample operations first
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
            # END BLOCK 1

            # BLOCK 2 [optional]: sample input feature blocks
            if self.with_input_blocks:
                next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                prev_c, prev_h = next_c, next_h
                hidden_states.append(prev_h)

                # block_query: num_blocks x 1
                block_query = tf.reshape(tf.matmul(next_h[-1], self.w_soft["input"]), (self.num_input_blocks, 1))
                # avoid multiple-connecting of input_blocks, if turned on
                if layer_id != self.num_layers - 1:
                    if self.input_block_unique_connection and layer_id > 0:
                        mask = tf.stop_gradient(tf.reduce_sum(tf.stack(input_block_record), axis=0))
                        mask = tf.reshape(mask, [self.num_input_blocks, 1])
                        mask1 = tf.greater(mask, 0)
                        block_query = tf.where(mask1, y=block_query, x=tf.fill(tf.shape(block_query), -10000.))
                else:
                    # added Oct.17.2019: make sure all inputs are connected to before last layer ends;
                    # i.e. get rid of default input nodes..
                    mask = tf.stop_gradient(tf.reduce_sum(tf.stack(input_block_record), axis=0))
                    mask = tf.reshape(mask, [self.num_input_blocks, 1])
                    mask2 = tf.equal(mask, 0)
                    block_query = tf.where(mask2, y=block_query, x=tf.fill(tf.shape(block_query), 10000.))

                # logit: tensor of shape = num_blocks x 2
                logit = tf.concat([-block_query, block_query], axis=1)
                if self.temperature is not None:
                    logit /= self.temperature
                if self.tanh_constant is not None:
                    logit = self.tanh_constant * tf.tanh(logit)
                probs_.append(tf.expand_dims(tf.nn.softmax(logit), axis=0))
                input_block = tf.multinomial(logit, 1)
                input_block = tf.to_int32(input_block)
                input_block = tf.reshape(input_block, [self.num_input_blocks])
                arc_seq.append(input_block)
                input_block_record.append(input_block)
                log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logit, labels=input_block)
                log_probs.append(tf.reshape(tf.reduce_sum(log_prob), [-1]))
                entropy = tf.stop_gradient(  # log_prob * tf.exp(-log_prob))
                    tf.reshape(tf.reduce_sum(log_prob * tf.exp(-log_prob)), [-1]))
                entropys.append(entropy)

                # removed Oct.1.2019; added back Oct.10.2019; modified to average Nov.19.2019
                # inputs: get a row slice of [out_filter[i]-1, lstm_size]
                inputs = tf.cast(tf.reshape(input_block, (1, self.num_input_blocks)), tf.float32)
                inputs /= (1.0 + tf.cast(tf.reduce_sum(input_block), tf.float32))
                inputs = tf.matmul(inputs, self.input_emb)
                # END BLOCK 2

            # BLOCK 3: sample the connections, unless the first layer
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
                        # comment out to disallow totally skipped layers
                        # if layer_id == self.num_layers - 1:
                        #    mask2 = tf.equal(mask, 0)
                        #    query = tf.where(mask2, y=query, x=tf.fill(tf.shape(query), 10000.))
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

            # END BLOCK 3

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

    # overwrite
    def _build_trainer(self):
        anchors = []
        anchors_w_1 = []
        probs_ = []

        ops_each_layer = 1
        total_arc_len = sum([ops_each_layer + self.num_input_blocks * self.with_input_blocks] + [
            ops_each_layer + self.num_input_blocks * self.with_input_blocks + i * self.with_skip_connection for i in
            range(1, self.num_layers)])
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
        inputs = tf.matmul(tf.ones((batch_size, 1)), self.g_emb)
        skip_targets = tf.constant([1.0 - self.skip_target, self.skip_target],
                                   dtype=tf.float32)

        arc_pointer = 0
        input_block_record = []
        skip_conn_record = []
        hidden_states = []
        for layer_id in range(self.num_layers):

            next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            prev_c, prev_h = next_c, next_h
            hidden_states.append(prev_h)

            logit = tf.matmul(next_h[-1], self.w_soft["start"][layer_id])  # batch_size x out_filter
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

            if self.with_input_blocks:

                next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                prev_c, prev_h = next_c, next_h
                hidden_states.append(prev_h)

                # next_h: batch_size x lstm_size; input: lstm_size x num_input_blocks
                block_query = tf.reshape(tf.matmul(next_h[-1], self.w_soft["input"]),
                                         (self.num_input_blocks * batch_size, 1))

                if layer_id != self.num_layers - 1:
                    if self.input_block_unique_connection and layer_id > 0:
                        mask = tf.stop_gradient(tf.reduce_sum(tf.stack(input_block_record), axis=0))
                        mask = tf.reshape(mask, [self.num_input_blocks * batch_size, 1])
                        mask1 = tf.greater(mask, 0)
                        block_query = tf.where(mask1, y=block_query, x=tf.fill(tf.shape(block_query), -10000.))
                else:
                    # added 10.17.2019: make sure all inputs are connected to before last layer ends;
                    # i.e. get rid of default input nodes..
                    mask = tf.stop_gradient(tf.reduce_sum(tf.stack(input_block_record), axis=0))
                    mask = tf.reshape(mask, [self.num_input_blocks * batch_size, 1])
                    mask2 = tf.equal(mask, 0)
                    block_query = tf.where(mask2, y=block_query, x=tf.fill(tf.shape(block_query), 10000.))

                logit = tf.concat([-block_query, block_query], axis=1)
                if self.temperature is not None:
                    logit /= self.temperature
                if self.tanh_constant is not None:
                    logit = self.tanh_constant * tf.tanh(logit)
                probs_.append(tf.reshape(tf.nn.softmax(logit), [batch_size, self.num_input_blocks, 2]))
                input_block = self.input_arc[
                              (arc_pointer + ops_each_layer): (arc_pointer + ops_each_layer + self.num_input_blocks)]
                input_block = tf.reshape(tf.transpose(input_block), [batch_size * self.num_input_blocks])
                input_block_record.append(input_block)
                # print("input_block", input_block)
                # print("logit", logit)
                log_prob2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logit, labels=input_block)
                log_prob2 = tf.reshape(log_prob2, [batch_size, -1])
                log_probs.append(tf.reduce_sum(log_prob2, axis=1))
                entropy = tf.stop_gradient(  # log_prob * tf.exp(-log_prob))
                    tf.reshape(tf.reduce_sum(log_prob2 * tf.exp(-log_prob2)), [-1]))
                entropys.append(entropy)

                # removed 10.1.2019; added back 10.10.2019
                # inputs: get a row slice of [out_filter[i]-1, lstm_size]
                inputs = tf.cast(tf.reshape(input_block, (batch_size, self.num_input_blocks)), tf.float32)
                inputs /= tf.matmul(
                    tf.reshape((1.0 + tf.cast(
                        tf.reduce_sum(
                            tf.reshape(input_block, (batch_size, self.num_input_blocks)),
                            axis=1), tf.float32)
                                ), (-1, 1)),
                    tf.ones((1, self.num_input_blocks), dtype=tf.float32))
                inputs = tf.matmul(inputs, self.input_emb)

            # sample the connections, unless the first layer
            # the number `skip` of each layer grows as layer_id grows
            if self.with_skip_connection:
                if layer_id > 0:

                    next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                    prev_c, prev_h = next_c, next_h
                    hidden_states.append(prev_h)

                    query = tf.transpose(tf.stack(anchors_w_1), [1, 0, 2])  # batch_size x layer_id x lstm_size
                    # print('query',query)
                    # print('next_h',next_h[-1])
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
                        # comment out to disallow totally skipped layers
                        # if layer_id == self.num_layers - 1:
                        #    mask2 = tf.equal(mask, 0)
                        #    query = tf.where(mask2, y=query, x=tf.fill(tf.shape(query), 10000.))
                    logit = tf.concat([-query, query], axis=1)  # logit: batch_size*layer_id x 2
                    if self.temperature is not None:
                        logit /= self.temperature
                    if self.tanh_constant is not None:
                        logit = self.tanh_constant * tf.tanh(logit)

                    probs_.append(tf.reshape(tf.nn.softmax(logit), [batch_size, layer_id, 2]))
                    if self.with_input_blocks:
                        skip = self.input_arc[(arc_pointer + ops_each_layer + self.num_input_blocks): (
                                arc_pointer + ops_each_layer + self.num_input_blocks + layer_id)]
                    else:
                        skip = self.input_arc[(arc_pointer + ops_each_layer): (arc_pointer + ops_each_layer + layer_id)]
                    # print(layer_id, (arc_pointer+2), (arc_pointer+2 + layer_id), skip)
                    skip = tf.reshape(tf.transpose(skip), [batch_size * layer_id])
                    skip = tf.to_int32(skip)

                    skip_prob = tf.sigmoid(logit)  # shape=(batch_size*layer_id, 2)
                    kl = skip_prob * tf.log(skip_prob / skip_targets)
                    kl = tf.reduce_sum(kl, axis=1)  # shape=(batch_size*layer_id,)
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

            arc_pointer += ops_each_layer + layer_id * self.with_skip_connection + \
                self.num_input_blocks * self.with_input_blocks
        # for DEBUG use
        self.train_hidden_states = hidden_states

        # for class attributes
        self.entropys = tf.stack(entropys)
        self.onehot_probs = probs_
        log_probs = tf.stack(log_probs)
        self.onehot_log_prob = tf.reduce_sum(log_probs, axis=0)
        skip_count = tf.stack(skip_count)
        self.onehot_skip_count = tf.reduce_sum(skip_count, axis=0)
        #skip_penaltys = tf.reduce_mean(tf.transpose(tf.stack(skip_penaltys), [1,0]), axis=1)
        skip_penaltys_flat = [tf.reduce_mean(x, axis=1) for x in skip_penaltys] # reduce_mean of layer_id dim; each entry's shape is  (batch_size,); in total list len=num_layers-1
        self.onehot_skip_penaltys = tf.reduce_mean(skip_penaltys_flat, axis=0)  # from [num_layers-1, batch_size] to [batch_size,]
        #self.onehot_skip_penaltys = tf.reduce_mean(skip_penaltys, axis=0)


class MultiIOController(MultiInputController):
    """
    Example
    ----------
    >>> from BioNAS.MockBlackBox.dense_skipcon_space import get_model_space
    >>> from BioNAS.Controller.multiio_controller import MultiIOController
    >>> import numpy as np
    >>> model_space = get_model_space(5)
    >>> controller = MultiIOController(model_space, output_block_unique_connection=True)
    >>> s = controller.session
    >>> a1, p1 = controller.get_action()
    >>> a2, p2 = controller.get_action()
    >>> a_batch = np.array([a1,a2])
    >>> p_batch = [np.concatenate(x) for x in zip(*[p1,p2])]
    >>> feed_dict = {controller.input_arc[i]: a_batch[:, [i]]
    >>>              for i in range(a_batch.shape[1])}
    >>> feed_dict.update({controller.advantage: np.array([1., -1.]).reshape((2,1))})
    >>> feed_dict.update({controller.old_probs[i]: p_batch[i]
    >>>                   for i in range(len(controller.old_probs))})
    >>> feed_dict.update({controller.reward: np.array([1., 1.]).reshape((2,1))})
    >>> print(s.run(controller.onehot_log_prob, feed_dict))
    >>> for _ in range(100):
    >>>     s.run(controller.train_op, feed_dict=feed_dict)
    >>>     if _%20==0: print(s.run(controller.loss, feed_dict))
    >>> print(s.run(controller.onehot_log_prob, feed_dict))

    Notes
    ----------
    Placeholder for now.
    """

    def __init__(self,
                 num_output_blocks=2,
                 with_output_blocks=True,
                 output_block_unique_connection=True,
                 output_block_diversity_weight=None,
                 **kwargs):

        # Attributes unique to the derived class:
        #self.with_input_blocks = True
        self.with_output_blocks = with_output_blocks
        self.num_output_blocks = num_output_blocks
        skip_weight = kwargs['skip_weight'] if 'skip_weight' in kwargs else None
        if output_block_diversity_weight is not None:
            assert skip_weight is not None, "Cannot use output_block_diversity_weight when skip_weight is None"
            self.output_block_diversity_weight = output_block_diversity_weight / skip_weight
        else:
            self.output_block_diversity_weight = None

        self.output_block_unique_connection = output_block_unique_connection
        super().__init__(**kwargs)
        assert self.with_skip_connection is True, "Must have with_skip_connection=True for MultiIOController"

    # override
    def _create_weight(self):
        super()._create_weight()
        with tf.variable_scope("outputs", initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)):
            self.w_soft['output'] = []
            for i in range(self.num_output_blocks):
                self.w_soft['output'].append(
                    tf.get_variable('output_block_%i' % i, [self.lstm_size, 1])
                )

    # override
    def _build_sampler(self):
        super()._build_sampler()
        step_size = 1 + int(self.with_input_blocks) + int(self.with_skip_connection)
        layer_hs = [self.sample_hidden_states[i][-1] for i in range(0, self.num_layers*step_size-1, step_size)]
        layer_hs = tf.concat(layer_hs, axis=0)
        output_probs = []
        output_onehot = []
        output_log_probs = []
        for i in range(self.num_output_blocks):
            logit = tf.matmul(layer_hs, self.w_soft['output'][i])
            if self.temperature is not None:
                logit /= self.temperature
            if self.tanh_constant is not None:
                logit = self.tanh_constant * tf.tanh(logit)
            if self.output_block_unique_connection:
                output_label = tf.reshape(tf.multinomial(tf.transpose(logit), 1), [-1])
                output = tf.one_hot(output_label, self.num_layers)
                prob = tf.nn.softmax(tf.squeeze(logit))
                prob = tf.reshape(prob, [1, -1])
                log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=tf.transpose(logit), labels=output_label)
            else:
                logit_ = tf.concat([-logit, logit], axis=1)
                output = tf.squeeze(tf.multinomial(logit_, 1))
                prob = tf.nn.sigmoid(logit_)
                prob = tf.reshape(prob, [1, -1, 2])
                log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logit_, labels=output)
            output_onehot.append(tf.cast(output, tf.int32))
            output_probs.append(prob)
            output_log_probs.append(log_prob)

        self.sample_probs.extend(output_probs)
        self.sample_arc = tf.concat([self.sample_arc, tf.reshape(output_onehot, [-1])], axis=0)
        self.sample_log_prob += tf.reduce_sum(output_log_probs)
        #if self.output_block_diversity_weight is not None:
        #    diversity = tf.math.reduce_std(output_probs, axis=0)
        #    diversity = tf.reduce_mean(diversity)
        #    self.skip_penaltys -= diversity * self.output_block_diversity_weight 

    # override
    def _build_trainer(self):
        super()._build_trainer()
        output_arc_len = self.num_layers * self.num_output_blocks
        self.input_arc += [tf.placeholder(shape=(None, 1), dtype=tf.int32, name='arc_{}'.format(i))
                           for i in range(self.total_arc_len, self.total_arc_len + output_arc_len)]
        self.total_arc_len += output_arc_len

        step_size = 1 + int(self.with_input_blocks) + int(self.with_skip_connection) 
        layer_hs = [self.train_hidden_states[i][-1] for i in range(0, self.num_layers*step_size-1, step_size)]
        layer_hs = tf.transpose(tf.stack(layer_hs), [1, 0, 2])  # shape: batch, num_layers, lstm_size
        output_probs = []
        output_log_probs = []
        for i in range(self.num_output_blocks):
            logit = tf.matmul(layer_hs, self.w_soft['output'][i])
            if self.temperature is not None:
                logit /= self.temperature
            if self.tanh_constant is not None:
                logit = self.tanh_constant * tf.tanh(logit)

            output = self.input_arc[-output_arc_len::][self.num_layers * i: self.num_layers * (i + 1)]
            output = tf.transpose(tf.squeeze(tf.stack(output), axis=-1))
            if self.output_block_unique_connection:
                logit = tf.transpose(logit, [0, 2, 1])
                prob = tf.squeeze(tf.nn.softmax(logit), axis=1)
                log_prob = tf.nn.softmax_cross_entropy_with_logits(
                    logits=logit, labels=output)
            else:
                logit_ = tf.concat([-logit, logit], axis=2)
                prob = tf.nn.sigmoid(logit_)
                log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logit_, labels=output)
                log_prob = tf.reshape(tf.reduce_mean(log_prob, axis=-1), [-1, 1])
            output_probs.append(prob)
            output_log_probs.append(log_prob)

        # for class attributes
        # NOTE: entropys is not added for the output_blocks
        self.onehot_probs.extend(output_probs)
        # self.output_log_probs = output_log_probs
        output_log_probs = tf.squeeze(tf.transpose(tf.stack(output_log_probs), [1, 0, 2]), axis=-1)
        self.onehot_log_prob += tf.reduce_sum(output_log_probs, axis=1)
        if self.output_block_diversity_weight is not None:
            output_probs = tf.transpose(tf.stack(output_probs), [1, 0, 2]) # shape: (batch, num_out_blocks, num_layers)
            diversity_penaltys = tf.math.reduce_std(output_probs, axis=1)  # std of probs. of out_blocks on each layer; connecting every out_block to one layer will be penalized
            diversity_penaltys = tf.reduce_mean(diversity_penaltys, axis=1)
            self.onehot_skip_penaltys -= diversity_penaltys * self.output_block_diversity_weight
