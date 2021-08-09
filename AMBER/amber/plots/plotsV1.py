# -*- coding: UTF-8 -*-

from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json

matplotlib.use('Agg')


def reset_style():
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['axes.titlesize'] = 14
    rcParams['axes.labelsize'] = 14
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.markersize'] = 8
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14
    rcParams['legend.fontsize'] = 14


reset_style()


def reset_plot(width_in_inches=4.5,
               height_in_inches=4.5):
    dots_per_inch = 200
    plt.close('all')
    return plt.figure(
        figsize=(width_in_inches, height_in_inches),
        dpi=dots_per_inch)


def rand_jitter(arr, scale=0.01):
    stdev = scale * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def heatscatter(x, y):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # plt.clf()
    reset_plot()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()


def heatscatter_sns(x, y, figsize=(8, 8)):
    sns.set(rc={'figure.figsize': figsize})
    sns.set(style="white", color_codes=True)
    sns.jointplot(x=x, y=y, kind='kde', color="skyblue")


def plot_training_history(history, par_dir):
    # print(history.history.keys())
    reset_plot()
    # summarize history for r2
    try:
        plt.plot(history.history['r_squared'])
        plt.plot(history.history['val_r_squared'])
        plt.title('model r2')
        plt.ylabel('r2')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        # plt.show()
        plt.savefig(os.path.join(par_dir, 'r2.png'))
        plt.gcf().clear()
    except:
        pass

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    plt.savefig(os.path.join(par_dir, 'loss.png'))
    plt.gcf().clear()


def plot_controller_performance(controller_hist_file, metrics_dict, save_fn=None, N_sma=10):
    '''
    Example:
        controller_hist_file = 'train_history.csv'
        metrics_dict = {'acc': 0, 'loss': 1, 'knowledge': 2}
    '''
    # plt.clf()
    reset_plot()
    plt.grid(b=True, linestyle='--', linewidth=0.8)
    df = pd.read_csv(controller_hist_file, header=None)
    assert df.shape[0] > N_sma
    df.columns = ['trial', 'loss_and_metrics', 'reward'] + ['layer_%i' % i for i in range(df.shape[1] - 3)]
    # N_sma = 20

    plot_idx = []
    for metric in metrics_dict:
        metric_idx = metrics_dict[metric]
        df[metric] = [float(x.strip('\[\]').split(',')[metric_idx]) for x in df['loss_and_metrics']]
        df[metric + '_SMA'] = np.concatenate(
            [[None] * (N_sma - 1), np.convolve(df[metric], np.ones((N_sma,)) / N_sma, mode='valid')])
        # df[metric+'_SMA'] /= np.max(df[metric+'_SMA'])
        plot_idx.append(metric + '_SMA')

    ax = sns.scatterplot(data=df[plot_idx])
    ax.set_xlabel('Steps')
    ax.set_ylabel('Simple Moving Average')
    if save_fn:
        plt.savefig(save_fn)
    else:
        plt.show()


def plot_environment_entropy(entropy_record, save_fn):
    '''plot the entropy change for the state-space
    in the training environment. A smaller entropy
    indicates converaged controller.
    '''
    # plt.clf()
    reset_plot()
    ax = sns.lineplot(np.arange(len(entropy_record)), entropy_record)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Entropy')
    plt.savefig(save_fn)


def sma(data, window=10):
    return np.concatenate([np.cumsum(data[:window - 1]) / np.arange(1, window),
                           np.convolve(data, np.ones((window,)) / window, mode='valid')])


def plot_action_weights(working_dir):
    save_path = os.path.join(working_dir, 'weight_data.json')
    if os.path.exists(save_path):
        with open(save_path, 'r+') as f:
            df = json.load(f)
        # plt.clf()
        for layer in df:
            reset_plot(width_in_inches=6, height_in_inches=4.5)
            ax = plt.subplot(111)
            for layer_name, data in df[layer]['operation'].items():
                d = np.array(data)
                if d.shape[0] >= 2:
                    avg = np.apply_along_axis(np.mean, 0, d)
                else:
                    avg = np.array(d).reshape(d.shape[1])
                ax.plot(avg, label=layer_name)
                if d.shape[0] >= 6:
                    std = np.apply_along_axis(np.std, 0, d) / np.sqrt(d.shape[0])
                    min_, max_ = avg - 1.96 * std, avg + 1.96 * std
                    ax.fill_between(range(avg.shape[0]), min_, max_, alpha=0.2)

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            ax.set_xlabel('Number of steps')
            ax.set_ylabel('Weight of layer type')
            # plt.title('Weight of Each Layer Type at Layer {}'.format(layer[-1]))
            plt.savefig(os.path.join(working_dir, 'weight_at_layer_{}.png'.format(layer.strip('L'))),
                        bbox_inches='tight')
    else:
        raise IOError('File does not exist')


def plot_wiring_weights(working_dir, with_input_blocks, with_skip_connection):
    if (not with_input_blocks) and (not with_skip_connection):
        return
    save_path = os.path.join(working_dir, 'weight_data.json')
    if os.path.exists(save_path):
        with open(save_path, 'r+') as f:
            df = json.load(f)
        for layer in df:
            # reset the plots, size and content
            reset_plot(width_in_inches=6, height_in_inches=4.5)
            if with_input_blocks:
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111)
                for layer_name, data in df[layer]['input_blocks'].items():
                    d = np.array(data)
                    if d.shape[0] >= 2:
                        avg = np.apply_along_axis(np.mean, 0, d)
                    else:
                        avg = np.array(d).reshape(d.shape[1])
                    ax1.plot(avg, label=layer_name)
                    if d.shape[0] >= 6:
                        std = np.apply_along_axis(np.std, 0, d) / np.sqrt(d.shape[0])
                        min_, max_ = avg - 1.96 * std, avg + 1.96 * std
                        ax1.fill_between(range(avg.shape[0]), min_, max_, alpha=0.2)

                box = ax1.get_position()
                ax1.set_position([box.x0, box.y0, box.width * 0.75, box.height])

                # Put a legend to the right of the current axis
                ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                ax1.set_xlabel('Number of steps')
                ax1.set_ylabel('Weight of input blocks')
                # plt.title('Weight of Each Layer Type at Layer {}'.format(layer[-1]))
                fig1.savefig(os.path.join(working_dir, 'inp_at_layer_{}.png'.format(layer.strip('L'))),
                             bbox_inches='tight')

            if with_skip_connection and int(layer.strip('L')) > 0:
                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111)
                for layer_name, data in df[layer]['skip_connection'].items():
                    d = np.array(data)
                    if d.shape[0] >= 2:
                        avg = np.apply_along_axis(np.mean, 0, d)
                    else:
                        avg = np.array(d).reshape(d.shape[1])
                    ax2.plot(avg, label=layer_name)
                    if d.shape[0] >= 6:
                        std = np.apply_along_axis(np.std, 0, d) / np.sqrt(d.shape[0])
                        min_, max_ = avg - 1.96 * std, avg + 1.96 * std
                        ax2.fill_between(range(avg.shape[0]), min_, max_, alpha=0.2)

                box = ax2.get_position()
                ax2.set_position([box.x0, box.y0, box.width * 0.75, box.height])

                # Put a legend to the right of the current axis
                ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                ax2.set_xlabel('Number of steps')
                ax2.set_ylabel('Weight of skip connection')
                fig2.savefig(os.path.join(working_dir, 'skip_at_layer_{}.png'.format(layer.strip('L'))),
                             bbox_inches='tight')
    else:
        raise IOError('File does not exist')


def plot_stats(working_dir):
    save_path = os.path.join(working_dir, 'nas_training_stats.json')
    if os.path.exists(save_path):
        df = json.load(open(save_path))
        # plt.clf()
        reset_plot()
        for item in ['Loss', 'Knowledge', 'Accuracy']:
            data = df[item]
            d = np.stack(list(map(lambda x: sma(x), np.array(data))), axis=0)
            avg = np.apply_along_axis(np.mean, 0, d)
            plt.plot(avg, label=item)
            if d.shape[0] >= 6:
                std = np.apply_along_axis(np.std, 0, d) / np.sqrt(d.shape[0])
                min_, max_ = avg - 1.96 * std, avg + 1.96 * std
                plt.fill_between(range(avg.shape[0]), min_, max_, alpha=0.2)

        plt.legend(loc='best')
        plt.xlabel('Number of steps')
        plt.ylabel('Statistics')
        # plt.title('Knowledge, Accuracy, and Loss over time')
        plt.savefig(os.path.join(working_dir, 'nas_training_stats.png'), bbox_inches='tight')
    else:
        raise IOError('File not found')


def plot_stats2(working_dir):
    save_path = os.path.join(working_dir, 'nas_training_stats.json')
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            df = json.load(f)
        # plt.clf()
        reset_plot()
        # ax = plt.subplot(111)
        data = df['Loss']
        d = np.stack(list(map(lambda x: sma(x), np.array(data))), axis=0)
        avg = np.apply_along_axis(np.mean, 0, d)
        ax = sns.lineplot(x=np.arange(1, len(avg) + 1), y=avg,
                          color='b', label='Loss', legend=False)
        if d.shape[0] >= 6:
            std = np.apply_along_axis(np.std, 0, d) / np.sqrt(d.shape[0])
            min_, max_ = avg - 1.96 * std, avg + 1.96 * std
            ax.fill_between(range(avg.shape[0]), min_, max_, alpha=0.2)

        ax2 = ax.twinx()
        data = df['Knowledge']
        d = np.stack(list(map(lambda x: sma(x), np.array(data))), axis=0)
        avg = np.apply_along_axis(np.mean, 0, d)
        sns.lineplot(x=np.arange(1, len(avg) + 1), y=avg,
                     color='g', label='Knowledge', ax=ax2, legend=False)
        if d.shape[0] >= 6:
            std = np.apply_along_axis(np.std, 0, d) / np.sqrt(d.shape[0])
            min_, max_ = avg - 1.96 * std, avg + 1.96 * std
            ax2.fill_between(range(avg.shape[0]), min_, max_, alpha=0.2)

        ax.figure.legend()
        ax.set_xlabel('Number of steps')
        ax.set_ylabel('Loss')
        ax2.set_ylabel('Knowledge')
        plt.savefig(os.path.join(working_dir, 'nas_training_stats.png'), bbox_inches='tight')
    else:
        raise IOError('File not found')


def accum_opt(data, find_min):
    tmp = []
    best = np.inf if find_min else -np.inf
    for d in data:
        if find_min and d < best:
            best = d
        elif (not find_min) and d > best:
            best = d
        tmp.append(best)
    return tmp


def multi_distplot_sns(data, labels, save_fn, title='title', xlab='xlab', ylab='ylab', hist=False, rug=False, xlim=None,
                       ylim=None, legend_off=False, **kwargs):
    assert len(data) == len(labels)
    # plt.clf()
    reset_plot()
    ax = sns.distplot(data[0], hist=hist, rug=rug, label=labels[0], **kwargs)
    if len(data) > 1:
        for i in range(1, len(data)):
            sns.distplot(data[i], hist=hist, rug=rug, label=labels[i], ax=ax, **kwargs)
    # ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc='upper left')
    if legend_off:
        ax.get_legend().remove()
    if save_fn:
        plt.savefig(save_fn)
    else:
        return ax


def violin_sns(data, x, y, hue, save_fn=None, split=True, **kwargs):
    # plt.clf()
    reset_plot()
    ax = sns.violinplot(x=x, y=y, hue=hue, split=split, data=data, **kwargs)
    if save_fn:
        plt.savefig(save_fn)
    else:
        return ax


def plot_controller_hidden_states(controller, save_fn='controller_hidden_states.png'):
    s = controller.session
    hidden_states, arc = s.run([controller.sample_hidden_states, controller.sample_arc])
    hidden_states_map = pd.DataFrame(np.concatenate([hidden_states[i][-1] for i in range(len(hidden_states))], axis=0))

    reset_plot(width_in_inches=hidden_states_map.shape[1] / 1.5, height_in_inches=hidden_states_map.shape[0] / 1.5)
    ax = sns.heatmap(np.round(hidden_states_map, 3), annot=True)

    # for a seaborn bug that cuts off top&bottom rows
    ax.set_ylim(len(hidden_states_map) - 0.5, -0.5)

    if save_fn:
        plt.savefig(save_fn)
    else:
        return ax


def plot_hessian(gkf, save_fn):
    h = np.apply_along_axis(np.mean, 0, gkf.W_model)
    h_var = np.apply_along_axis(np.var, 0, gkf.W_model)

    plt.clf()
    # change the label to mean, and cell-size to represent variance
    vmax = np.max(h) * 1.5
    vmin = np.min(h) * 1.5
    cellsize_vmax = np.max(h_var) * 1.2
    g_ratio = h
    g_size = h_var
    annot = np.vectorize(lambda x: "" if np.isnan(x) else "{:.2f}".format(x))(g_ratio)

    # adjust visual balance
    figsize = (g_ratio.shape[1], g_ratio.shape[0])
    cbar_width = 0.02 * 6.0 / figsize[0]

    f, ax = plt.subplots(1, 1, figsize=figsize)
    cbar_ax = f.add_axes([0.9, 0.1, cbar_width, 0.8])
    cbar_ax.tick_params(length=0.5, labelsize='x-small')
    
    from .heatmap2 import heatmap2
    heatmap2(g_ratio, ax=ax,
             cbar_ax=cbar_ax,
             vmax=vmax, vmin=vmin,
             cmap='Spectral_r',
             annot=annot, fmt="s", annot_kws={"fontsize": "medium"},
             cellsize=g_size, cellsize_vmax=cellsize_vmax,
             square=True, ax_kws={"title": "Model Hessian"})
    f.savefig(save_fn)


def plot_sequence_importance(model_importance,
                             motif_importance,
                             pos_chunks=None,
                             neg_chunks=None,
                             prediction=None,
                             seq_char=None,
                             title='sequence_importance',
                             linespace=0.025,
                             save_fn=None
                             ):
    def normalizer(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    seq_len = len(model_importance)
    fig = reset_plot(width_in_inches=seq_len / 1000 * 9, height_in_inches=4.5)
    ax = fig.add_subplot(111)
    # ax.plot(linespace + normalizer(model_importance), color='r', label='Model Saliency')
    # ax.plot(linespace + normalizer(model_importance), color='r', label='Model Saliency')
    lns1 = ax.plot(model_importance, color='r', label='Model Saliency')
    # ax.plot(-(linespace + normalizer(motif_importance)), color='b', label='Motif Score')
    # ax.plot(-(linespace + motif_importance), color='b', label='Motif Score')
    ax2 = ax.twinx()
    lns2 = ax2.plot(motif_importance, color='b', label='Motif Score')
    # cannot afford to put seq_char in for now; ZZ 2019.12.11
    if seq_char is not None:
        ax.text(0, 0, seq_char)
        # ax.set_xticks(np.arange(len(seq_char)), list(seq_char))
    if pos_chunks is not None:
        chunk = np.stack(pos_chunks)
        ax.hlines(y=np.zeros(chunk.shape[0]), xmin=chunk[:, 0], xmax=chunk[:, 1], color='orange', linewidth=8,
                  label='Important Region')
    if neg_chunks is not None:
        chunk = np.stack(neg_chunks)
        ax.hlines(y=np.zeros(chunk.shape[0]), xmin=chunk[:, 0], xmax=chunk[:, 1], color='grey', linewidth=8,
                  label='Nuisance Region')

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    # ax.legend(loc='upper left')
    # ax.set_ylim((-1.05-linespace, 1.05+linespace))
    ax.set_xlabel('Nucleotide Position')
    ax.set_ylabel('Normalized Score')
    if prediction is None:
        ax.set_title(title)
    else:
        ax.set_title(title + ', pred=%s' % prediction)

    if save_fn is not None:
        plt.tight_layout()
        plt.savefig(save_fn)
