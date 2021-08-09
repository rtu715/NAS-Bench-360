# -*- coding: UTF-8 -*-
"""Manager class for streamlining downstream build and evaluation given an architecture.

Manager is the class that takes in architecture designs from an architecture search/optimization algorithm, then
interacts with ``amber.modeler`` to build and train the model according to architecture, and finally calls
``amber.architect.rewards`` to evaluate the trained model rewards to feedback the architecture designer.

"""

import gc
import os, sys
import warnings

import numpy as np
import tensorflow.keras as keras
from ..utils import corrected_tf as tf
import tensorflow as tf2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
import time
from datetime import datetime
from collections import defaultdict

from .commonOps import unpack_data
from .store import get_store_fn

__all__ = [
    'BaseNetworkManager',
    'NetworkManager',
    'GeneralManager',
    'DistributedGeneralManager'
]


class BaseNetworkManager:
    def __init__(self, *args, **kwargs):
        # abstract
        pass

    def get_rewards(self, trial, model_arc):
        raise NotImplementedError("Abstract method.")


class GeneralManager(BaseNetworkManager):
    """Manager creates child networks, train them on a dataset, and retrieve rewards.

    Parameters
    ----------
    train_data : tuple, string or generator
        Training data to be fed to ``keras.models.Model.fit``.

    validation_data : tuple, string, or generator
        Validation data. The data format is understood similarly to train_data.

    model_fn : amber.modeler
        A callable function to build and implement child models given an architecture sequence.

    reward_fn : amber.architect.rewards
        A callable function to evaluate the rewards on a trained model and the validation dataset.

    store_fn : amber.architect.store
        A callable function to store necessary information (such as predictions, model architectures, and a variety of
        plots etc.) for the given child model.

    working_dir : str
        File path for working directory.

    save_full_model : bool
        If true, save the full model beside the model weights. Default is False.

    epochs : int
        The total number of epochs to train the child model.

    child_batchsize : int
        The batch size for training the child model.

    fit_kwargs : dict or None
        Keyword arguments for model.fit

    predict_kwargs : dict or None
        Keyword arguments for model.predict

    evaluate_kwargs : dict or None
        Keyword arguments for model.evaluate

    verbose : bool or int
        Verbose level. 0=non-verbose, 1=verbose, 2=less verbose.

    kwargs : dict
        Other keyword arguments parsed.


    Attributes
    ----------
    train_data : tuple or generator
        The unpacked training data

    validation_data : tuple or generator
        The unpacked validation data

    model_fn : amber.modeler
        Reference to the callable function to build and implement child models given an architecture sequence.

    reward_fn : amber.architect.rewards
        Reference to the callable function to evaluate the rewards on a trained model and the validation dataset.

    store_fn : amber.architect.store
        Reference to the callable function to store necessary information (such as predictions, model architectures, and a variety of
        plots etc.) for the given child model.

    working_dir : str
        File path to working directory

    verbose : bool or int
        Verbose level

    TODO
    ------
    - Refactor the rest of attributes as private.
    - Update the description of ``train_data``  and ``validation_data`` to more flexible unpacking, once it's added::

        If it's tuple, expects it to be a tuple of numpy.array of
        (x,y); if it's string, expects it to be the file path to a compiled training data; if it's a generator, expects
        it yield a batch of training features and samples.

    """

    def __init__(self,
                 train_data,
                 validation_data,
                 model_fn,
                 reward_fn,
                 store_fn,
                 working_dir='.',
                 save_full_model=False,
                 epochs=5,
                 child_batchsize=128,
                 verbose=0,
                 fit_kwargs=None,
                 predict_kwargs=None,
                 evaluate_kwargs=None,
                 **kwargs):
        super(GeneralManager, self).__init__(**kwargs)
        self.train_data = train_data
        self.validation_data = validation_data
        self.working_dir = working_dir
        self.fit_kwargs = fit_kwargs or {}
        self.predict_kwargs = predict_kwargs or {}
        self.evaluate_kwargs = evaluate_kwargs or {}
        self._earlystop_patience = self.fit_kwargs.pop("earlystop_patience",5)
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        self.model_compile_dict = kwargs.pop("model_compile_dict", None)
        if self.model_compile_dict is None:
            self.model_compile_dict = model_fn.model_compile_dict

        # added 2020.5.19: parse model_space to manager for compatibility with newer versions of controllers
        self.model_space = kwargs.pop("model_space", None)

        self.save_full_model = save_full_model
        self.epochs = epochs
        self.batchsize = child_batchsize
        self.verbose = verbose

        self.model_fn = model_fn
        self.reward_fn = reward_fn
        self.store_fn = get_store_fn(store_fn)

    def get_rewards(self, trial, model_arc, **kwargs):
        """The reward getter for a given model architecture

        Parameters
        ----------
        trial : int
            An integer number indicating the trial for this architecture

        model_arc : list
            The list of architecture sequence

        Returns
        -------
        this_reward : float
            The reward signal as determined by ``reward_fn(model, val_data)``

        loss_and_metrics : dict
            A dictionary of auxillary information for this model, such as loss, and other metrics (as in ``tf.keras.metrics``)
        """
        # print('-'*80, model_arc, '-'*80)
        train_graph = tf.Graph()
        train_sess = tf.Session(graph=train_graph)
        with train_graph.as_default(), train_sess.as_default():
            try:
                K.set_session(train_sess)
            except RuntimeError: # keras 2.3.1 `set_session` not available for tf2.0
                assert keras.__version__ > '2.2.5'
                pass
            model = self.model_fn(model_arc)  # a compiled keras Model
            if model is None:
                assert hasattr(self.reward_fn, "min"), "model_fn of type %s returned a non-valid model, but the given " \
                                                       "reward_fn of type %s does not have .min() method" % (type(
                    self.model_fn), type(self.reward_fn))
                hist = None
                this_reward, loss_and_metrics, reward_metrics = self.reward_fn.min(data=self.validation_data)
                loss = loss_and_metrics.pop(0)
                loss_and_metrics = {str(self.model_compile_dict['metrics'][i]): loss_and_metrics[i] for i in
                                    range(len(loss_and_metrics))}
                loss_and_metrics['loss'] = loss
                if reward_metrics:
                    loss_and_metrics.update(reward_metrics)
            else:
                # train the model using Keras methods
                if self.verbose:
                    print(" Trial %i: Start training model..." % trial)
                train_x, train_y = unpack_data(self.train_data)
                hist = model.fit(x=train_x,
                                 y=train_y,
                                 batch_size=self.batchsize if train_y is not None else None,
                                 epochs=self.epochs,
                                 verbose=self.verbose,
                                 #shuffle=True,
                                 validation_data=self.validation_data,
                                 callbacks=[ModelCheckpoint(os.path.join(self.working_dir, 'temp_network.h5'),
                                                            monitor='val_loss', verbose=self.verbose,
                                                            save_best_only=True),
                                            EarlyStopping(monitor='val_loss', patience=self.fit_kwargs.pop("earlystop_patience", 5), verbose=self.verbose)],
                                 **self.fit_kwargs
                                 )
                # load best performance epoch in this training session
                # in corner cases, the optimization might fail and no temp_network 
                # would be created
                if os.path.isfile((os.path.join(self.working_dir, 'temp_network.h5'))):
                    model.load_weights(os.path.join(self.working_dir, 'temp_network.h5'))
                else:
                    model.save_weights((os.path.join(self.working_dir, 'temp_network.h5')))

                # evaluate the model by `reward_fn`
                this_reward, loss_and_metrics, reward_metrics = \
                    self.reward_fn(model, self.validation_data,
                                   session=train_sess,
                                   graph=train_graph)
                loss = loss_and_metrics.pop(0)
                loss_and_metrics = {str(self.model_compile_dict['metrics'][i]): loss_and_metrics[i] for i in
                                    range(len(loss_and_metrics))}
                loss_and_metrics['loss'] = loss
                if reward_metrics:
                    loss_and_metrics.update(reward_metrics)

                # do any post processing,
                # e.g. save child net, plot training history, plot scattered prediction.
                if self.store_fn:
                    val_pred = model.predict(self.validation_data, verbose=self.verbose, **self.predict_kwargs)
                    self.store_fn(
                        trial=trial,
                        model=model,
                        hist=hist,
                        data=self.validation_data,
                        pred=val_pred,
                        loss_and_metrics=loss_and_metrics,
                        working_dir=self.working_dir,
                        save_full_model=self.save_full_model,
                        knowledge_func=self.reward_fn.knowledge_function
                    )

        # clean up resources and GPU memory
        del model
        del hist
        gc.collect()
        return this_reward, loss_and_metrics


class DistributedGeneralManager(GeneralManager):
    """Distributed manager will place all tensors of any child models to a pre-assigned GPU device
    """
    def __init__(self, devices, train_data_kwargs, validate_data_kwargs, do_resample=False, *args, **kwargs):
        self.devices = devices
        super().__init__(*args, **kwargs)
        assert devices is None or len(self.devices) == 1, "Only supports one GPU device currently"
        # For keeping & closing file connection at multi-processing
        self.train_data_kwargs = train_data_kwargs or {}
        self.validate_data_kwargs = validate_data_kwargs or {}
        self.train_x = None
        self.train_y = None
        self.file_connected = False
        # For resampling; TODO: how to implement a Bayesian version of this?
        self.arc_records = defaultdict(dict)
        self.do_resample = do_resample

    def close_handler(self):
        if self.file_connected:
            self.train_x.close()
            if self.train_y:
                self.train_y.close()
            self._validation_data_gen.close()
            self.train_x = None
            self.train_y = None
            self.file_connected = False

    def get_rewards(self, trial, model_arc, remap_device=None, **kwargs):
        # TODO: use tensorflow distributed strategy
        #strategy = tf2.distribute.MirroredStrategy(devices=self.devices)
        #print('Number of devices: {} - {}'.format(strategy.num_replicas_in_sync, self.devices))
        #with strategy.scope():
        pid = os.getpid()
        sys.stderr.write("[%s][%s] Preprocessing.."%(pid, datetime.now().strftime("%H:%M:%S") ))
        start_time = time.time()
        train_graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        train_sess = tf.Session(graph=train_graph, config=config)
        # remap device will overwrite the manager device
        if remap_device is not None:
            target_device = remap_device
        elif self.devices is None:
            from ..utils.gpu_query import get_idle_gpus
            idle_gpus = get_idle_gpus()
            target_device = idle_gpus[0]
            target_device = "/device:GPU:%i"%target_device
            self.devices = [target_device]
            sys.stderr.write("[%s] Auto-assign device: %s" % (pid, target_device) )
        else:
            target_device = self.devices[0]
        with train_graph.as_default(), train_sess.as_default():
            with tf.device(target_device):
                try:
                    K.set_session(train_sess)
                except RuntimeError: # keras 2.3.1 `set_session` not available for tf2.0
                    pass
                model = self.model_fn(model_arc)  # a compiled keras Model

                # unpack the dataset
                if not self.file_connected:
                    X_train, y_train = unpack_data(self.train_data, callable_kwargs=self.train_data_kwargs)
                    self.train_x = X_train
                    self.train_y = y_train
                    assert callable(self.validation_data), "Expect validation_data to be callable, got %s" % type(self.validation_data)
                    self._validation_data_gen = self.validation_data(**self.validate_data_kwargs)
                    self.file_connected = True
                elapse_time = time.time() - start_time
                sys.stderr.write("  %.3f sec\n"%elapse_time)
                model_arc_ = tuple(model_arc)
                if model_arc_ in self.arc_records and self.do_resample is True:
                    this_reward = self.arc_records[model_arc_]['reward']
                    old_trial = self.arc_records[model_arc_]['trial']
                    loss_and_metrics = self.arc_records[model_arc_]['loss_and_metrics']
                    sys.stderr.write("[%s][%s] Trial %i: Re-sampled from history %i\n" % (pid, datetime.now().strftime("%H:%M:%S"), trial, old_trial))
                else:
                    # train the model using Keras methods
                    start_time = time.time()
                    sys.stderr.write("[%s][%s] Trial %i: Start training model.." % (pid, datetime.now().strftime("%H:%M:%S"), trial))
                    hist = model.fit(self.train_x, self.train_y,
                                     batch_size=self.batchsize,
                                     epochs=self.epochs,
                                     verbose=self.verbose,
                                     validation_data=self._validation_data_gen,
                                     callbacks=[ModelCheckpoint(os.path.join(self.working_dir, 'temp_network.h5'),
                                                                monitor='val_loss', verbose=self.verbose,
                                                                save_best_only=True),
                                                EarlyStopping(monitor='val_loss', patience=self._earlystop_patience, verbose=self.verbose)],
                                     **self.fit_kwargs
                                     )

                    # load best performance epoch in this training session
                    model.load_weights(os.path.join(self.working_dir, 'temp_network.h5'))
                    elapse_time = time.time() - start_time
                    sys.stderr.write("  %.3f sec\n"%elapse_time)

                    start_time = time.time()
                    sys.stderr.write("[%s] Postprocessing.."% pid )
                    # evaluate the model by `reward_fn`
                    this_reward, loss_and_metrics, reward_metrics = \
                        self.reward_fn(model, self._validation_data_gen,
                                       session=train_sess,
                                       )
                    loss = loss_and_metrics.pop(0)
                    loss_and_metrics = {str(self.model_compile_dict['metrics'][i]): loss_and_metrics[i] for i in
                                        range(len(loss_and_metrics))}
                    loss_and_metrics['loss'] = loss
                    if reward_metrics:
                        loss_and_metrics.update(reward_metrics)

                    # do any post processing,
                    # e.g. save child net, plot training history, plot scattered prediction.
                    if self.store_fn:
                        val_pred = model.predict(self.validation_data, verbose=self.verbose)
                        self.store_fn(
                            trial=trial,
                            model=model,
                            hist=hist,
                            data=self._validation_data_gen,
                            pred=val_pred,
                            loss_and_metrics=loss_and_metrics,
                            working_dir=self.working_dir,
                            save_full_model=self.save_full_model,
                            knowledge_func=self.reward_fn.knowledge_function
                        )
                    elapse_time = time.time() - start_time
                    sys.stderr.write("  %.3f sec\n"%elapse_time)

                    # store the rewards in records
                    self.arc_records[model_arc_]['trial'] = trial
                    self.arc_records[model_arc_]['reward'] = this_reward
                    self.arc_records[model_arc_]['loss_and_metrics'] = loss_and_metrics

        # clean up resources and GPU memory
        start_time = time.time()
        sys.stderr.write("[%s] Cleaning up.."%pid)
        try:
            del train_sess
            del train_graph
            del model
            del hist
        except UnboundLocalError:
            pass
        gc.collect()
        elapse_time = time.time() - start_time
        sys.stderr.write("  %.3f sec\n"%elapse_time)
        return this_reward, loss_and_metrics


class EnasManager(GeneralManager):
    """A specialized manager for Efficient Neural Architecture Search (ENAS).

    Because

    Parameters
    ----------
    session : tensorflow.Session or None
        The tensorflow session that the manager will be parsed to modelers. By default it's None, which will then get the
        Session from the modeler.

    train_data : tuple, string or generator
        Training data to be fed to ``keras.models.Model.fit``.

    validation_data : tuple, string, or generator
        Validation data. The data format is understood similarly to train_data.

    model_fn : amber.modeler
        A callable function to build and implement child models given an architecture sequence. Must be a model_fn that
        is compatible with ENAS parameter sharing.

    reward_fn : amber.architect.rewards
        A callable function to evaluate the rewards on a trained model and the validation dataset.

    store_fn : amber.architect.store
        A callable function to store necessary information (such as predictions, model architectures, and a variety of
        plots etc.) for the given child model.

    working_dir : str
        File path for working directory.

    Attributes
    ----------
    model : amber.modeler.child
        The child DAG that is connected to ``controller.sample_arc`` as the input architecture sequence, which
        will activate a randomly sampled subgraph within child DAG. Because it's hard-wired to the sampled architecture
        in controller, using this model to train and predict will also have the inherent stochastic behaviour that is
        linked to controller.

        See Also
        --------
        amber.modeler.child : AMBER wrapped-up version of child models that is intended to have similar interface and
            methods as the ``keras.models.Model`` API.

    train_data : tuple or generator
        The unpacked training data

    validation_data : tuple or generator
        The unpacked validation data

    model_fn : amber.modeler
        Reference to the callable function to build and implement child models given an architecture sequence.

    reward_fn : amber.architect.rewards
        Reference to the callable function to evaluate the rewards on a trained model and the validation dataset.

    store_fn : amber.architect.store
        Reference to the callable function to store necessary information (such as predictions, model architectures, and a variety of
        plots etc.) for the given child model.

    disable_controller : bool
        If true, will randomly return a reward by uniformly sampling in the interval [0,1]. Default is False.

    working_dir : str
        File path to working directory

    verbose : bool or int
        Verbose level

    """
    def __init__(self, session=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if session is None:
            self.session = self.model_fn.session
        else:
            self.session = session
        self.model = None
        self.disable_controller = kwargs.pop("disable_controller", False)

    def get_rewards(self, trial, model_arc=None, nsteps=None):
        """The reward getter for a given model architecture.

        Because Enas will train child model by random sampling an architecture to activate for each mini-batch,
        there will not be any rewards evaluation in the Manager anymore.
        However, we can still use `get_rewards` as a proxy to train child models

        Parameters
        ----------
        trial : int
            An integer number indicating the trial for this architecture

        model_arc : list or None
            The list of architecture sequence. If is None (as by default), will return the child DAG with architecture
            connected directly to ``controller.sample_arc`` tensors.


        nsteps: int
            Optional, if specified, train model nsteps of batches instead of a whole epoch

        Returns
        -------
        this_reward : float
            The reward signal as determined by ``reward_fn(model, val_data)``

        loss_and_metrics : dict
            A dictionary of auxillary information for this model, such as loss, and other metrics (as in ``tf.keras.metrics``)


        """
        if self.model is None:
            self.model = self.model_fn()

        if model_arc is None:
            # unpack the dataset
            X_val, y_val = self.validation_data[0:2]
            X_train, y_train = self.train_data
            # train the model using EnasModel methods
            if self.verbose:
                print(" Trial %i: Start training model with sample_arc..." % trial)
            hist = self.model.fit(X_train, y_train,
                                  batch_size=self.batchsize,
                                  nsteps=nsteps,
                                  epochs=self.epochs,
                                  verbose=self.verbose,
                                  # comment out because of temporary
                                  # incompatibility with tf.data.Dataset
                                  # validation_data=(X_val, y_val),
                                  )

            # do any post processing,
            # e.g. save child net, plot training history, plot scattered prediction.
            if self.store_fn:
                val_pred = self.model.predict(X_val, verbose=self.verbose)
                self.store_fn(
                    trial=trial,
                    model=self.model,
                    hist=hist,
                    data=self.validation_data,
                    pred=val_pred,
                    loss_and_metrics=None,
                    working_dir=self.working_dir,
                    save_full_model=self.save_full_model,
                    knowledge_func=self.reward_fn.knowledge_function
                )
            return None, None
        else:
            model = self.model_fn(model_arc)
            this_reward, loss_and_metrics, reward_metrics = \
                self.reward_fn(model, self.validation_data,
                               session=self.session)
            loss = loss_and_metrics.pop(0)
            loss_and_metrics = {str(self.model_compile_dict['metrics'][i]): loss_and_metrics[i] for i in
                                range(len(loss_and_metrics))}
            loss_and_metrics['loss'] = loss
            if reward_metrics:
                loss_and_metrics.update(reward_metrics)
            # enable this to overwrite a random reward when disable controller
            if self.disable_controller:
                this_reward = np.random.uniform(0, 1)
            # end
            return this_reward, loss_and_metrics

