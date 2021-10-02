import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K
import importlib
import gc


class History(object):
    """
    Custom class to help get log data from keras.callbacks.History objects.
    :param history: a ``keras.callbacks.History object`` or ``None``.
    """
    def __init__(self, history=None):
        if history is not None:
            self.epoch = history.epoch
            self.history = history.history
        else:
            self.epoch = []
            self.history = {}


def concatenate_history(hlist, reindex_epoch=False):
    """
    A helper function to concatenate training history object (``keras.callbacks.History``) into a single one, with a help ``History`` class.
    :param hlist: a list of ``keras.callbacks.History`` objects to concatenate.
    :param reindex_epoch: True or False whether to reindex epoch counters to an increasing order.
    :return his: an instance of ``History`` class that contain concatenated information of epoch and training history.
    """

    his = History()

    for h in hlist:
        his.epoch = his.epoch + h.epoch

        for key, value in h.history.items():
            his.history.setdefault(key, [])
            his.history[key] = his.history[key] + value

    if reindex_epoch:
        his.epoch = list(np.arange(0, len(his.epoch)))

    return his


def plot_from_history(history):
    """
    Plot losses in training history.
    :param history: a ``keras.callbacks.History`` or (this module's) ``History`` object.
    """
    assert isinstance(history, (keras.callbacks.History, History)), "history must be a ``keras.callbacks.History`` or " \
                                                                    "(this module's) ``History`` object. "

    epoch = history.epoch
    val_exist = "val_loss" in history.history

    plt.plot(epoch, history.history["loss"], '.-', label="train")
    if val_exist:
        plt.plot(epoch, history.history["val_loss"], '.-', label="valid")

    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend()


def save_history_to_csv(history, filepath):
    """
    Save a training history into a csv file.
    :param history: a ``History`` callback instance from ``Model`` instance.
    :param filepath: a string filepath.
    """
    hist = history.history
    hist['epoch'] = history.epoch
    df = pd.DataFrame.from_dict(hist)
    df.to_csv(filepath, index=False)


def reset_keras(per_process_gpu_memory_fraction=1.0):
    """
    Reset Keras session and set GPU configuration as well as collect unused memory.
    This is adapted from [jaycangel's post on fastai forum](https://forums.fast.ai/t/how-could-i-release-gpu-memory-of-keras/2023/18).
    Calling this before any training will clear Keras session. Hence, a Keras model must be redefined and compiled again.
    It can be used in during hyperparameter scan or K-fold validation when model training is invoked several times.
    :param per_process_gpu_memory_fraction: tensorflow's config.gpu_options.per_process_gpu_memory_fraction
    """
    sess = K.get_session()
    K.clear_session()
    sess.close()

    gc.collect()

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    config.gpu_options.visible_device_list = "0"
    K.set_session(tf.Session(config=config))


def cuda_release_memory():
    """
    Force cuda to release GPU memory by closing it.
    :return cuda: numba's cuda module.
    """
    spec = importlib.util.find_spec("numba")
    if spec is None:
        raise Exception("numba module cannot be found. Can't function before numba module is installed.")
    else:
        from numba import cuda
    cuda.select_device(0)
    cuda.close()
    return cuda


def moving_window_avg(x, window=5):
    """
    Return a moving-window average.
    :param x: a numpy array
    :param window: an integer, number of data points for window size.
    """
    return pd.DataFrame(x).rolling(window=window, min_periods=1).mean().values.squeeze()


def set_momentum(optimizer, mom_val):
    """
    Helper to set momentum of Keras optimizers.
    :param optimizer: Keras optimizer
    :param mom_val: value of momentum.
    """
    keys = dir(optimizer)
    if "momentum" in keys:
        K.set_value(optimizer.momentum, mom_val)
    if "rho" in keys:
        K.set_value(optimizer.rho, mom_val)
    if "beta_1" in keys:
        K.set_value(optimizer.beta_1, mom_val)


def set_lr(optimizer, lr):
    """
    Helper to set learning rate of Keras optimizers.
    :param optimizer: Keras optimizer
    :param lr: value of learning rate.
    """
    K.set_value(optimizer.lr, lr)
