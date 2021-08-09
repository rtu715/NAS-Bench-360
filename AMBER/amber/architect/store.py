# -*- coding: UTF-8 -*-

"""
Store functions will take care of storage and post-processing related matters after a child model is trained.
"""

import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from ..plots import plot_hessian, plot_training_history
from .commonOps import unpack_data


def get_store_fn(arg):
    """The getter function that returns a callable store function from a string

    Parameters
    ----------
    arg : str
        The string identifier for a particular store function. Current choices are:
        - general
        - model_plot
        - minimal

    Returns
    -------
    callable
        A callable store function
    """
    if callable(arg) is True:
        return arg
    elif arg is None:
        return None
    elif arg.lower() == 'store_general' or arg.lower() == 'general':
        return store_general
    elif arg.lower() == 'store_regression' or arg.lower() == 'regression':
        return store_regression
    elif arg.lower() == 'store_with_hessian' or arg.lower() == 'hessian':
        return store_with_hessian
    elif arg.lower() == 'store_with_model_plot' or arg.lower() == 'model_plot':
        return store_with_model_plot
    elif arg.lower() == 'store_minimal' or arg.lower() == 'minimal':
        return store_minimal
    else:
        raise Exception('cannot understand store_fn: %s' % arg)


def store_with_model_plot(
        trial,
        model,
        hist,
        data,
        pred,
        loss_and_metrics,
        working_dir='.',
        save_full_model=False,
        *args, **kwargs
):
    par_dir = os.path.join(working_dir, 'weights', 'trial_%s' % trial)
    os.makedirs(par_dir, exist_ok=True)
    store_general(trial=trial,
                  model=model,
                  hist=hist,
                  data=data,
                  pred=pred,
                  loss_and_metrics=loss_and_metrics,
                  working_dir=working_dir,
                  save_full_model=save_full_model
                  )
    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file=os.path.join(par_dir, "model_arc.png"), show_shapes=True, show_layer_names=True)


def store_with_hessian(
        trial,
        model,
        hist,
        data,
        pred,
        loss_and_metrics,
        working_dir='.',
        save_full_model=False,
        knowledge_func=None
):
    assert knowledge_func is not None, "`store_with_hessian` requires parsing the" \
                                       "knowledge function used."
    par_dir = os.path.join(working_dir, 'weights', 'trial_%s' % trial)
    store_general(trial=trial,
                  model=model,
                  hist=hist,
                  data=data,
                  pred=pred,
                  loss_and_metrics=loss_and_metrics,
                  working_dir=working_dir,
                  save_full_model=save_full_model
                  )
    # plot_training_history(hist, par_dir)
    from keras.utils import plot_model
    plot_model(model, to_file=os.path.join(par_dir, "model_arc.png"))

    plot_hessian(knowledge_func, os.path.join(par_dir, "hess.png"))
    knowledge_func._reset()
    plt.close('all')
    return


def store_general(
        trial,
        model,
        hist,
        data,
        pred,
        loss_and_metrics,
        working_dir='.',
        save_full_model=False,
        *args, **kwargs
):
    par_dir = os.path.join(working_dir, 'weights', 'trial_%s' % trial)
    if os.path.isdir(par_dir):
        shutil.rmtree(par_dir)
    os.makedirs(par_dir)
    if save_full_model:
        model.save(os.path.join(working_dir, 'weights', 'trial_%s' % trial, 'full_bestmodel.h5'))
    try:
        plot_training_history(hist, par_dir)
    except:
        # TODO: this still not working for non-keras models, e.g. EnasAnn/Cnn
        pass
    if os.path.isfile(os.path.join(working_dir, 'temp_network.h5')):
        shutil.move(os.path.join(working_dir, 'temp_network.h5'), os.path.join(par_dir, 'bestmodel.h5'))
    data = unpack_data(data, unroll_generator_y=True)
    metadata = data[2] if len(data) > 2 else None
    obs = data[1]
    write_pred_to_disk(
        os.path.join(par_dir, 'pred.txt'),
        pred, obs, metadata,
        loss_and_metrics
    )


def store_minimal(
        trial,
        model,
        working_dir='.',
        save_full_model=False,
        **kwargs
):
    par_dir = os.path.join(working_dir, 'weights', 'trial_%s' % trial)
    if os.path.isdir(par_dir):
        shutil.rmtree(par_dir)
    os.makedirs(par_dir)
    if save_full_model:
        model.save(os.path.join(working_dir, 'weights', 'trial_%s' % trial, 'full_bestmodel.h5'))
    if os.path.isfile(os.path.join(working_dir, 'temp_network.h5')):
        shutil.move(os.path.join(working_dir, 'temp_network.h5'), os.path.join(par_dir, 'bestmodel.h5'))


def write_pred_to_disk(fn, y_pred, y_obs, metadata=None, metrics=None):
    with open(fn, 'w') as f:
        if metrics is not None:
            f.write('\n'.join(['# {}: {}'.format(x, metrics[x]) for x in metrics]) + '\n')
        if type(y_pred) is list:
            y_pred = np.concatenate(y_pred, axis=1)
            y_obs = np.concatenate(y_obs, axis=1)
        if len(np.unique(y_obs)) < 10:  # is categorical
            str_format = "%i"
        else:
            str_format = "%.3f"

        f.write('pred\tobs\tmetadata\n')
        for i in range(len(y_pred)):
            if len(y_pred[i].shape) > 1 or y_pred[i].shape[0] > 1:
                y_pred_i = ','.join(['%.3f' % x for x in np.array(y_pred[i])])
                y_obs_i = ','.join([str_format % x for x in np.array(y_obs[i])])
            else:
                y_pred_i = '%.3f' % y_pred[i]
                y_obs_i = str_format % y_obs[i]
            if metadata:
                f.write('%s\t%s\t%s\n' % (y_pred_i, y_obs_i, metadata[i]))
            else:
                f.write('%s\t%s\t%s\n' % (y_pred_i, y_obs_i, 'NA'))
