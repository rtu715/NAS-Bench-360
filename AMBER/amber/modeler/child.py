# -*- coding: UTF-8 -*-

"""
Child model classes wrapped above Keras.Model API for more complex child
network manipulations
"""

# Author: ZZJ
# Initial date: 10.1.2019


import datetime
import warnings
from tqdm import tqdm
import h5py
import numpy as np
import tensorflow as tf
from keras.callbacks import CallbackList, BaseLogger, History
from keras.models import Model
from keras.utils.data_utils import GeneratorEnqueuer
from tqdm import trange

from ..architect.commonOps import batchify, numpy_shuffle_in_unison


class GeneralChild(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DenseAddOutputChild(GeneralChild):
    def __init__(self, nodes=None, block_loss_mapping=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.outputs) > 1:
            # self.aux_loss_weight = 0.25 / (len(self.outputs)-1)
            self.aux_loss_weight = 0.1
        else:
            self.aux_loss_weight = 0
        self.nodes = nodes
        self.block_loss_mapping = block_loss_mapping

    def _expand_label(self, y):
        if type(y) is not list:
            y_ = [y] * len(self.outputs)
        else:
            assert len(y) == len(self.outputs), "if `y` is provided as list, it has to match the added " \
                                                "output dimension; got len(y)=%i but len(outputs)=%i" % (
                                                    len(y), len(self.outputs))
            y_ = y
        return y_

    def compile(self, *args, **kwargs):
        return super().compile(*args, **kwargs,
                               loss_weights=[1.] + [self.aux_loss_weight] * (len(self.outputs) - 1))

    def fit(self, x, y, *args, **kwargs):
        if 'validation_data' in kwargs:
            kwargs['validation_data'] = list(kwargs['validation_data'])
            kwargs['validation_data'][1] = self._expand_label(kwargs['validation_data'][1])
        y_ = self._expand_label(y)
        return super().fit(x=x, y=y_, *args, **kwargs)

    def evaluate(self, x, y, final_only=True, *args, **kwargs):
        y_ = self._expand_label(y)
        # the loss and metrics are distributed as
        # total_loss, loss_0 (output), loss_1 (added_out1), loss_2 (added_out2), ..
        # metrics_0 (output), metrics_1 (added_out1), ..
        loss_and_metrics = super().evaluate(x, y_)
        if final_only and len(self.outputs) > 1:
            metrics = [
                loss_and_metrics[(len(self.outputs) + 1):][i] for i in
                range(0, len(loss_and_metrics[(len(self.outputs) + 1):]), len(self.outputs))
            ]
            loss = loss_and_metrics[1]
            return [loss] + metrics
        else:
            return loss_and_metrics

    def predict(self, x, final_only=True, *args, **kwargs):
        if final_only:
            y_pred = super().predict(x)
            if len(self.outputs) > 1:
                return y_pred[0]
            else:
                return y_pred
        else:
            return super().predict(x)


class EnasAnnModel:
    def __init__(self, inputs, outputs, arc_seq, dag, session, dropouts=None, name='EnasModel'):
        """
        Parameters
        ----------
            inputs: tf.Tensor
                input tensors/placeholders
            outputs: tf.Tensor
                output tensors
            session: tf.Session
                tensorflow Session for use
            name: str
                name for tf.variable_scope; default is "EnasDAG"
        """
        assert type(inputs) in (tf.Tensor, list), "get unexpected inputs types: %s" % type(inputs)
        assert type(outputs) in (tf.Tensor, list), "get unexpected outputs types: %s" % type(outputs)
        self.arc_seq = arc_seq
        self.dag = dag
        self.inputs = [inputs] if type(inputs) is tf.Tensor else inputs
        self.outputs = [outputs] if type(outputs) is tf.Tensor else outputs
        labels = dag.child_model_label
        self.labels = [labels] if type(labels) is tf.Tensor else labels
        self.session = session
        self.dropouts = dropouts
        self.dropout_placeholders = None
        self.name = name
        self.trainable_var = [var for var in tf.trainable_variables() if var.name.startswith(self.name)]
        self.train_op = None
        self.optimizer = None
        self.optimizer_ = None
        self.lr = None
        self.grad_norm = None
        self.loss = None
        self.metrics = None
        self.weights = None
        self.loss_weights = None
        self.is_compiled = False
        self.nodes = None
        self.use_pipe = None
        self.reinitialize_train_pipe = None

        # for Keras
        self.stop_training = False

    def compile(self, optimizer, loss=None, metrics=None, loss_weights=None):
        assert not self.is_compiled, "already compiled"
        if self.arc_seq is None:
            if self.dag.train_fixed_arc:
                self.train_op = self.dag.fixed_train_op
                self.optimizer = self.dag.fixed_optimizer
                self.loss = self.dag.fixed_loss
                self.metrics = self.dag.fixed_metrics
                self.weights = self.dag.fixed_w_masks
                # self.loss_weights = self.dag.loss_weights
                self.dropout_placeholders = self.dag.fixed_dropouts

            else:
                self.train_op = self.dag.sample_train_op
                self.optimizer = self.dag.sample_optimizer
                self.loss = self.dag.sample_loss
                self.metrics = self.dag.sample_metrics
                self.weights = self.dag.sample_w_masks
                # self.loss_weights = self.dag.loss_weights
                self.dropout_placeholders = self.dag.sample_dropouts
        else:
            self.train_op = self.dag.fixed_train_op
            self.optimizer = self.dag.fixed_optimizer
            self.loss = self.dag.fixed_loss
            self.metrics = self.dag.fixed_metrics
            self.weights = self.dag.fixed_w_masks
            # self.loss_weights = self.dag.loss_weights
            self.dropout_placeholders = self.dag.fixed_dropouts

        if self.dropouts:
            assert len(self.dropout_placeholders) == len(self.dropouts), "provided dropout probs of len %i does not " \
                                                                         "match number of layers: %i" % (
                                                                         len(self.dropout_placeholders),
                                                                         len(self.dropouts))

        if self.arc_seq is None and self.dag.feature_model is not None and self.dag.feature_model.pseudo_inputs_pipe is not None:
            self.use_pipe = True
            self.reinitialize_train_pipe = True
        else:
            self.use_pipe = False

        self.is_compiled = True
        return

    def _make_feed_dict(self, x=None, y=None, is_training_phase=False):
        assert x is None or type(x) is list, "x arg for _make_feed_dict must be List"
        assert y is None or type(y) is list, "x arg for _make_feed_dict must be List"
        if self.arc_seq is None:
            feed_dict = {}
        else:
            feed_dict = {self.dag.input_arc[i]: self.arc_seq[i] for i in range(len(self.arc_seq))}
        if x is not None:
            for i in range(len(self.inputs)):
                if len(x[i].shape) > 1:
                    feed_dict.update({self.inputs[i]: x[i]})
                else:
                    feed_dict.update({self.inputs[i]: x[i][np.newaxis, :]})
        if y is not None:
            for i in range(len(self.outputs)):
                if len(y[i].shape) > 1:
                    feed_dict.update({self.labels[i]: y[i]})
                else:
                    feed_dict.update({self.labels[i]: np.expand_dims(y[i], -1)})
        if is_training_phase and self.dropouts:
            feed_dict.update({self.dropout_placeholders[i]: self.dropouts[i]
                              for i in range(len(self.dropouts))})
        return feed_dict

    def _make_tf_dataset(self, x_, y_=None, shuffle=False):
        assert type(x_) is list, "x arg for _make_tf_dataset must be List"
        assert y_ is None or type(y_) is list, "x arg for _make_tf_dataset must be List"
        if shuffle:
            if y_ is None:
                print("shuffling x")
                numpy_shuffle_in_unison(x_)
            else:
                print("shuffling x and y")
                numpy_shuffle_in_unison(x_ + y_)
        feature_model = self.dag.feature_model
        data_pipe_feed = {feature_model.x_ph[i]: x_[i] for i in range(len(x_))}
        if y_ is None:
            total_len = len(x_[0])
            y_ = [np.zeros((total_len,) + tuple(i.value for i in self.outputs[i].shape[1:]))
                  for i in range(len(self.outputs))]
        data_pipe_feed.update({feature_model.y_ph[i]: y_[i] for i in range(len(y_))})
        self.session.run(feature_model.data_gen.initializer,
                         feed_dict=data_pipe_feed)

    def fit(self, x, y, batch_size=None, nsteps=None, epochs=1, verbose=1, callbacks=None, validation_data=None):
        if self.use_pipe:
            return self.fit_pipe(
                x=x,
                y=y,
                batch_size=batch_size,
                nsteps=nsteps,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_data=validation_data)
        else:
            return self.fit_ph(
                x=x,
                y=y,
                batch_size=batch_size,
                nsteps=nsteps,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_data=validation_data)

    def fit_generator(self,
                      generator,
                      steps_per_epoch=None,
                      epochs=1,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      shuffle=True):
        if workers > 0:
            enqueuer = GeneratorEnqueuer(
                generator,
                use_multiprocessing=use_multiprocessing)

            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            output_generator = generator

        callback_list = CallbackList(callbacks=callbacks)
        callback_list.set_model(self)
        callback_list.on_train_begin()

        hist = {'loss': [], 'val_loss': []}
        for epoch in range(epochs):
            seen = 0
            epoch_logs = {'loss': 0, 'val_loss': 0}
            t = trange(steps_per_epoch) if verbose == 1 else range(steps_per_epoch)
            for _ in t:
                generator_output = next(output_generator)
                x, y = generator_output
                if x is None or len(x) == 0:
                    # Handle data tensors support when no input given
                    # step-size = 1 for data tensors
                    batch_size = 1
                elif isinstance(x, list):
                    batch_size = x[0].shape[0]
                elif isinstance(x, dict):
                    batch_size = list(x.values())[0].shape[0]
                else:
                    batch_size = x.shape[0]
                batch_loss, batch_metrics = self.train_on_batch(x, y)
                epoch_logs['loss'] += batch_loss * batch_size
                seen += batch_size

            for k in epoch_logs:
                epoch_logs[k] /= seen
            hist['loss'].append(epoch_logs['loss'])

            if validation_data:
                val_loss_and_metrics = self.evaluate(validation_data[0], validation_data[1])
                hist['val_loss'].append(val_loss_and_metrics[0])
                epoch_logs.update({'val_loss': val_loss_and_metrics[0]})

            callback_list.on_epoch_end(epoch, epoch_logs)

            if self.stop_training:
                break
        if workers > 0:
            enqueuer.stop()
        return hist

    def train_on_batch(self, x, y):
        feed_dict = self._make_feed_dict(x, y, is_training_phase=True)
        _, batch_loss, batch_metrics = self.session.run([self.train_op, self.loss, self.metrics], feed_dict=feed_dict)
        return batch_loss, batch_metrics

    def evaluate(self, *args, **kwargs):
        if self.use_pipe:
            return self.evaluate_pipe(*args, **kwargs)
        else:
            return self.evaluate_ph(*args, **kwargs)

    def predict(self, *args, **kwargs):
        if self.use_pipe:
            # print('='*80); print('predict with pipe')
            return self.predict_pipe(*args, **kwargs)
        else:
            # print('='*80); print('predict with placeholder')
            return self.predict_ph(*args, **kwargs)

    def fit_pipe(self, x, y, batch_size=None, nsteps=None, epochs=1, verbose=1, callbacks=None, validation_data=None):
        hist = {'loss': [], 'val_loss': []}
        assert epochs > 0
        assert self.dag.feature_model is not None
        feature_model = self.dag.feature_model
        if batch_size is None:
            batch_size = feature_model.batch_size
        # overwrite
        # batch_size = feature_model.batch_size
        total_len = len(y[0]) if type(y) is list else len(y)
        nsteps = total_len // batch_size
        if type(x) is list:
            x_ = x
        else:
            x_ = [x]
        if type(y) is list:
            y_ = y
        else:
            y_ = [y]
        for epoch in range(epochs):
            if self.reinitialize_train_pipe:
                self._make_tf_dataset(x_, y_)
                self.reinitialize_train_pipe = False
            t = trange(nsteps) if verbose == 1 else range(nsteps)
            metrics_val = []
            curr_loss = None
            for _ in t:
                try:
                    feed_dict = self._make_feed_dict()
                    _, batch_loss, batch_metrics = self.session.run([self.train_op, self.loss, self.metrics],
                                                                    feed_dict=feed_dict)
                    if len(metrics_val):
                        metrics_val = list(map(lambda x: x[0] * 0.95 + x[1] * 0.05, zip(metrics_val, batch_metrics)))
                    else:
                        metrics_val = batch_metrics
                    curr_loss = batch_loss if curr_loss is None else curr_loss * 0.95 + batch_loss * 0.05
                    if verbose == 1:
                        t.set_postfix(loss="%.4f" % curr_loss)
                except tf.errors.OutOfRangeError:
                    self.reinitialize_train_pipe = True
                    warnings.warn("train pipe out of range")
                    break

            hist['loss'].append(curr_loss)
            if validation_data:
                val_loss_and_metrics = self.evaluate(validation_data[0], validation_data[1])
                hist['val_loss'].append(val_loss_and_metrics[0])

            if verbose:
                if validation_data:
                    print("Epoch %i, loss=%.3f, metrics=%s; val=%s" % (
                    epoch, curr_loss, metrics_val, val_loss_and_metrics))
                else:
                    print("Epoch %i, loss=%.3f, metrics=%s" % (epoch, curr_loss, metrics_val))
        return hist

    def fit_ph(self, x, y, batch_size=None, nsteps=None, epochs=1, verbose=1, callbacks=None, validation_data=None):
        hist = {'loss': [], 'val_loss': []}
        total_len = len(y[0]) if type(y) is list else len(y)
        if nsteps is None:
            nsteps = total_len // batch_size
        callback_list = CallbackList(callbacks=callbacks)
        callback_list.set_model(self)
        callback_list.on_train_begin()
        assert epochs > 0
        g = batchify(x, y, batch_size)
        for epoch in range(epochs):
            t = trange(nsteps) if verbose == 1 else range(nsteps)
            metrics_val = []
            curr_loss = None
            for it in t:
                try:
                    x_, y_ = next(g)
                except StopIteration:
                    g = batchify(x, y, batch_size)
                    x_, y_ = next(g)
                feed_dict = self._make_feed_dict(x_, y_, is_training_phase=True)
                _, batch_loss, batch_metrics = self.session.run([self.train_op, self.loss, self.metrics],
                                                                feed_dict=feed_dict)
                if len(metrics_val):
                    metrics_val = list(map(lambda x: x[0] * 0.95 + x[1] * 0.05, zip(metrics_val, batch_metrics)))
                else:
                    metrics_val = batch_metrics
                curr_loss = batch_loss if curr_loss is None else curr_loss * 0.95 + batch_loss * 0.05
                if verbose == 1:
                    t.set_postfix(loss="%.4f" % curr_loss)
                if verbose == 2:
                    if it % 1000 == 0:
                        print(
                        "%s %i/%i, loss=%.5f" % (datetime.datetime.now().strftime("%H:%M:%S"), it, nsteps, curr_loss),
                        flush=True)

            hist['loss'].append(curr_loss)
            logs = {'loss': curr_loss}
            if validation_data:
                val_loss_and_metrics = self.evaluate(validation_data[0], validation_data[1])
                hist['val_loss'].append(val_loss_and_metrics[0])
                logs.update({'val_loss': val_loss_and_metrics[0]})

            if verbose:
                if validation_data:
                    print("Epoch %i, loss=%.3f, metrics=%s; val=%s" % (
                    epoch, curr_loss, metrics_val, val_loss_and_metrics))
                else:
                    print("Epoch %i, loss=%.3f, metrics=%s" % (epoch, curr_loss, metrics_val))

            callback_list.on_epoch_end(epoch=epoch, logs=logs)
            if self.stop_training:
                break
        return hist

    def predict_ph(self, x, batch_size=None):
        if type(x) is not list: x = [x]
        if batch_size is None:
            batch_size = min(1000, len(x[0]))
        y_pred_ = []
        for x_ in batchify(x, None, batch_size=batch_size, shuffle=False, drop_remainder=False):
            feed_dict = self._make_feed_dict(x_)
            y_pred = self.session.run(self.outputs, feed_dict)
            y_pred_.append(y_pred)
        y_pred = [np.concatenate(t, axis=0) for t in zip(*y_pred_)]
        if len(y_pred) > 1:
            y_pred = [y for y in y_pred]
        else:
            y_pred = y_pred[0]
        return y_pred

    def predict_pipe(self, x, batch_size=None, verbose=0):
        total_len = len(x[0]) if type(x) is list else len(x)
        if type(x) is not list:
            x_ = [x]
        else:
            x_ = x
        feature_model = self.dag.feature_model
        # if batch_size is None:
        #    batch_size = feature_model.batch_size
        # overwrite
        batch_size = feature_model.batch_size
        y_pred_ = []
        self._make_tf_dataset(x_)
        nsteps = total_len // batch_size
        t = trange(nsteps) if verbose else range(nsteps)
        for _ in t:
            feed_dict = self._make_feed_dict()
            y_pred = self.session.run(self.outputs, feed_dict)
            y_pred_.append(y_pred)
        y_pred = [np.concatenate(t, axis=0) for t in zip(*y_pred_)]
        if len(y_pred) > 1:
            y_pred = [y for y in y_pred]
        else:
            y_pred = y_pred[0]
        return y_pred

    def evaluate_ph(self, x, y, batch_size=None, verbose=0):
        if batch_size is None:
            batch_size = min(100, x.shape[0])
        loss_and_metrics = []
        seen = 0
        if verbose:
            gen = tqdm(batchify(x, y, batch_size=batch_size, shuffle=False))
        else:
            gen = batchify(x, y, batch_size=batch_size, shuffle=False)
        for x_, y_ in gen:
            feed_dict = self._make_feed_dict(x_, y_)
            loss, metrics = self.session.run([self.loss, self.metrics], feed_dict=feed_dict)
            this_batch_size = x_[0].shape[0]
            if not len(loss_and_metrics):
                loss_and_metrics = [loss * this_batch_size] + [x * this_batch_size for x in metrics]
            else:
                tmp = [loss] + metrics
                loss_and_metrics = [loss_and_metrics[i] + this_batch_size * tmp[i] for i in range(len(tmp))]
            seen += this_batch_size
        loss_and_metrics = [x / seen for x in loss_and_metrics]
        return loss_and_metrics

    def evaluate_pipe(self, x, y, batch_size=None, verbose=0):
        loss_and_metrics = []
        feature_model = self.dag.feature_model
        total_len = len(y[0]) if type(y) is list else len(y)
        # if batch_size is None:
        #    batch_size = feature_model.batch_size
        # overwrite
        batch_size = feature_model.batch_size
        nsteps = total_len // batch_size
        if type(x) is list:
            x_ = x
        else:
            x_ = [x]
        if type(y) is list:
            y_ = y
        else:
            y_ = [y]
        self._make_tf_dataset(x_, y_)
        t = trange(nsteps) if verbose else range(nsteps)
        for _ in t:
            feed_dict = self._make_feed_dict()
            loss, metrics = self.session.run([self.loss, self.metrics], feed_dict=feed_dict)
            if not len(loss_and_metrics):
                loss_and_metrics = [loss] + metrics
            else:
                tmp = [loss] + metrics
                loss_and_metrics = [0.95 * loss_and_metrics[i] + 0.05 * tmp[i] for i in range(len(tmp))]
        return loss_and_metrics

    def save(self, *args, **kwargs):
        """
        TODO
        -----
            save model architectures
        """
        warnings.warn("Not implemented yet; rolling back to `save_weights`")
        self.save_weights(*args, **kwargs)
        return

    def save_weights(self, filepath, **kwargs):
        weights = self.get_weights()
        with h5py.File(filepath, 'w') as hf:
            for i, d in enumerate(weights):
                hf.create_dataset(name='layer_{:d}/weight'.format(i), data=d[0])
                hf.create_dataset(name='layer_{:d}/bias'.format(i), data=d[1])

    def load_weights(self, filepath, **kwargs):
        total_layers = len(self.weights)
        weights = []
        with h5py.File(filepath, 'r') as hf:
            for i in range(total_layers):
                w_key = "layer_{:d}/weight".format(i)
                b_key = "layer_{:d}/bias".format(i)
                weights.append((hf.get(w_key).value, hf.get(b_key).value))
        self.set_weights(weights)

    def get_weights(self, **kwargs):
        weights = self.session.run(self.weights, feed_dict=self._make_feed_dict())
        return weights

    def set_weights(self, weights, **kwargs):
        assign_ops = []
        weights_vars = list(zip(self.dag.w, self.dag.b)) + list(zip(self.dag.w_out, self.dag.b_out))
        for i in range(len(weights)):
            # weight
            assign_ops.append(tf.assign(weights_vars[i][0], weights[i][0]))
            # bias
            assign_ops.append(tf.assign(weights_vars[i][1], weights[i][1]))
        self.session.run(assign_ops)


class EnasCnnModel:
    """
    TODO
    -----
    - re-write weights save/load
    - use the input/output/label tensors provided by EnasConv1dDAG; this should unify the
      fit method when using placeholder and Tensor pipelines - probably still need two separate
      methods though

    """

    def __init__(self, inputs, outputs, labels, arc_seq, dag, session, dropouts=None, use_pipe=None, name='EnasModel',
                 **kwargs):
        assert type(inputs) in (tf.Tensor, list), "get unexpected inputs types: %s" % type(inputs)
        assert type(outputs) in (tf.Tensor, list), "get unexpected outputs types: %s" % type(outputs)
        self.arc_seq = arc_seq
        self.dag = dag
        self.inputs = [inputs] if type(inputs) is tf.Tensor else inputs
        self.outputs = [outputs] if type(outputs) is tf.Tensor else outputs
        self.callbacks = None
        self.labels = [labels] if type(labels) is tf.Tensor else labels
        self.session = session
        self.dropouts = dropouts
        self.dropout_placeholders = None
        self.name = name
        self.trainable_var = [var for var in tf.trainable_variables() if var.name.startswith(self.name)]
        self.train_op = None
        self.optimizer = None
        self.optimizer_ = None
        self.lr = None
        self.grad_norm = None
        self.loss = None
        self.metrics = None
        self.weights = None
        self.loss_weights = None
        self.is_compiled = False

        self.use_pipe = use_pipe or False
        self.reinitialize_train_pipe = None

        # added 2020.5.17: add default feed-dict for sampled architecture, to account for data description NAS
        self.sample_dag_feed_dict = kwargs.pop("sample_dag_feed_dict", {})

        # for Keras
        self.stop_training = False
        self.metrics_name = []
        self.batch_size = self.dag.batch_size

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None):
        assert not self.is_compiled, "already compiled"
        if self.arc_seq is None:
            if self.dag.train_fixed_arc:
                self.train_op = self.dag.fixed_train_op
                self.optimizer = self.dag.fixed_optimizer
                self.loss = self.dag.fixed_loss
                self.metrics = self.dag.fixed_metrics
                # self.loss_weights = self.dag.loss_weights
                self.dropout_placeholders = self.dag.fixed_dropouts

            else:
                self.train_op = self.dag.sample_train_op
                self.optimizer = self.dag.sample_optimizer
                self.loss = self.dag.sample_loss
                self.metrics = self.dag.sample_metrics
                # self.loss_weights = self.dag.loss_weights
                self.dropout_placeholders = self.dag.sample_dropouts
        else:
            self.train_op = self.dag.fixed_train_op
            self.optimizer = self.dag.fixed_optimizer
            self.loss = self.dag.fixed_loss
            self.metrics = self.dag.fixed_metrics
            # self.loss_weights = self.dag.loss_weights
            self.dropout_placeholders = self.dag.fixed_dropouts

        if self.dropouts:
            assert len(self.dropout_placeholders) == len(self.dropouts), "provided dropout probs of len %i does not " \
                                                                         "match number of layers: %i" % (
                                                                         len(self.dropout_placeholders),
                                                                         len(self.dropouts))
        if self.use_pipe:
            self.reinitialize_train_pipe = True
        if metrics:
            metrics = [metrics] if type(metrics) is not list else metrics
            self.metrics_name = [str(m) for m in metrics]
        self.weights = [v for v in tf.trainable_variables() if v.name.startswith(self.name)]
        self.is_compiled = True

    def _make_feed_dict(self, x=None, y=None, is_training_phase=False):
        assert x is None or type(x) is list, "x arg for _make_feed_dict must be List, got %s" % type(x)
        assert y is None or type(y) is list, "x arg for _make_feed_dict must be List, got %s" % type(y)
        if self.arc_seq is None:
            feed_dict = self.sample_dag_feed_dict
        else:
            feed_dict = {self.dag.input_arc[i]: self.arc_seq[i] for i in range(len(self.arc_seq))}
        if x is not None:
            for i in range(len(self.inputs)):
                if len(x[i].shape) > 1:
                    feed_dict.update({self.inputs[i]: x[i]})
                else:
                    feed_dict.update({self.inputs[i]: x[i][np.newaxis, :]})
        if y is not None:
            for i in range(len(self.outputs)):
                if len(y[i].shape) > 1:
                    feed_dict.update({self.labels[i]: y[i]})
                else:
                    feed_dict.update({self.labels[i]: np.expand_dims(y[i], -1)})
        if is_training_phase and self.dropouts:
            feed_dict.update({self.dropout_placeholders[i]: self.dropouts[i]
                              for i in range(len(self.dropouts))})
        elif (not is_training_phase) and len(self.dropout_placeholders):
            feed_dict.update({self.dropout_placeholders[i]: 0.0
                              for i in range(len(self.dropout_placeholders))})

        return feed_dict

    def fit(self, x, y, batch_size=None, nsteps=None, epochs=1, verbose=1, callbacks=None, validation_data=None):
        assert self.is_compiled, "Must compile model first"
        assert epochs > 0
        x = x if type(x) is list else [x]
        y = y if type(y) is list else [y]
        if nsteps is None:
            total_len = len(y[0]) if type(y) is list else len(y)
            nsteps = total_len // batch_size
        # BaseLogger should always be the first metric since it computes the stats on epoch end
        base_logger = BaseLogger(stateful_metrics=["val_%s" % m for m in self.metrics_name] + ['val_loss', 'size'])
        base_logger_params = {'metrics': ['loss'] + self.metrics_name}
        if validation_data:
            base_logger_params['metrics'] += ['val_%s' % m for m in base_logger_params['metrics']]
        base_logger.set_params(base_logger_params)
        hist = History()
        if callbacks is None:
            callbacks = [base_logger] + [hist]
        elif type(callbacks) is list:
            callbacks = [base_logger] + callbacks + [hist]
        else:
            callbacks = [base_logger] + [callbacks] + [hist]
        callback_list = CallbackList(callbacks=callbacks)
        callback_list.set_model(self)
        callback_list.on_train_begin()
        self.callbacks = callback_list
        for epoch in range(epochs):
            g = batchify(x, y, batch_size) if batch_size else None
            t = trange(nsteps) if verbose == 1 else range(nsteps)
            callback_list.on_epoch_begin(epoch)
            for it in t:
                x_, y_ = next(g) if g else (None, None)
                batch_logs = self.train_on_batch(x_, y_)
                callback_list.on_batch_end(it, batch_logs)
                curr_loss = base_logger.totals['loss'] / base_logger.seen
                if verbose == 1:
                    t.set_postfix(loss="%.4f" % curr_loss)
                if verbose == 2:
                    if it % 1000 == 0:
                        print(
                            "%s %i/%i, loss=%.5f" %
                            (datetime.datetime.now().strftime("%H:%M:%S"), it, nsteps, curr_loss),
                            flush=True)

            if validation_data:
                val_logs = self.evaluate(validation_data[0], validation_data[1])
                base_logger.on_batch_end(None, val_logs)

            epoch_logs = {}
            callback_list.on_epoch_end(epoch=epoch, logs=epoch_logs)

            if verbose:
                if validation_data:
                    to_print = ['loss'] + self.metrics_name + ['val_loss'] + ['val_%s' % m for m in self.metrics_name]
                else:
                    to_print = ['loss'] + self.metrics_name
                prog = ", ".join(["%s=%.4f" % (name, hist.history[name][-1]) for name in to_print])
                print("Epoch %i, %s" % (epoch, prog), flush=True)

            if self.stop_training:
                break

        return hist.history

    def fit_generator(self):
        pass

    def train_on_batch(self, x=None, y=None):
        assert self.is_compiled, "Must compile model first"
        feed_dict = self._make_feed_dict(x, y, is_training_phase=True)
        batch_size = x[0].shape[0] if x is not None else self.batch_size
        _, batch_loss, batch_metrics = self.session.run([self.train_op, self.loss, self.metrics], feed_dict=feed_dict)
        logs = {self.metrics_name[i]: batch_metrics[i] for i in range(len(self.metrics_name))}
        logs.update({'loss': batch_loss, 'size': batch_size})
        return logs

    def evaluate(self, x, y, batch_size=None, verbose=0):
        assert self.is_compiled, "Must compile model first"
        batch_size = batch_size or self.batch_size
        loss_and_metrics = []
        seen = 0
        gen = tqdm(batchify(x, y, batch_size=batch_size, shuffle=False, drop_remainder=False)) if verbose else \
            batchify(x, y, batch_size=batch_size, shuffle=False, drop_remainder=False)
        for x_, y_ in gen:
            feed_dict = self._make_feed_dict(x_, y_)
            loss, metrics = self.session.run([self.loss, self.metrics], feed_dict=feed_dict)
            this_batch_size = x_[0].shape[0]
            if not len(loss_and_metrics):
                loss_and_metrics = [loss * this_batch_size] + [x * this_batch_size for x in metrics]
            else:
                tmp = [loss] + metrics
                loss_and_metrics = [loss_and_metrics[i] + this_batch_size * tmp[i] for i in range(len(tmp))]
            seen += this_batch_size
        loss_and_metrics = [x / seen for x in loss_and_metrics]
        # return loss_and_metrics
        logs = {'val_loss': loss_and_metrics.pop(0)}
        logs.update({'val_%s' % self.metrics_name[i]: loss_and_metrics[i] for i in range(len(self.metrics_name))})
        return logs

    def predict(self, x, batch_size=None, verbose=0):
        assert self.is_compiled, "Must compile model first"
        if type(x) is not list: x = [x]
        batch_size = batch_size or self.batch_size
        y_pred_ = []
        if verbose:
            gen = tqdm(batchify(x, None, batch_size=batch_size, shuffle=False, drop_remainder=False))
        else:
            gen = batchify(x, None, batch_size=batch_size, shuffle=False, drop_remainder=False)
        for x_ in gen:
            feed_dict = self._make_feed_dict(x_)
            y_pred = self.session.run(self.outputs, feed_dict)
            y_pred_.append(y_pred)
        y_pred = [np.concatenate(t, axis=0) for t in zip(*y_pred_)]
        if len(y_pred) > 1:
            y_pred = [y for y in y_pred]
        else:
            y_pred = y_pred[0]
        return y_pred

    def save(self, *args, **kwargs):
        """
        TODO
        ------
            save model architectures
        """
        warnings.warn("Not implemented yet; rolling back to `save_weights`", stacklevel=2)
        self.save_weights(*args, **kwargs)
        return

    def save_weights(self, filepath, **kwargs):
        weights = self.get_weights()
        with h5py.File(filepath, 'w') as hf:
            for i, d in enumerate(weights):
                hf.create_dataset(name=self.weights[i].name, data=d)

    def load_weights(self, filepath, **kwargs):
        weights = []
        with h5py.File(filepath, 'r') as hf:
            for i in range(len(self.weights)):
                key = self.weights[i].name
                weights.append(hf.get(key).value)
        self.set_weights(weights)

    def get_weights(self, **kwargs):
        weights = self.session.run(self.weights, feed_dict=self._make_feed_dict())
        return weights

    def set_weights(self, weights, **kwargs):
        assign_ops = []
        for i in range(len(self.weights)):
            assign_ops.append(tf.assign(self.weights[i], weights[i]))
        self.session.run(assign_ops)
