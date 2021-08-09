import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from amber.plots._plotsV1 import sma


def num_of_val_pos(wd):
    managers = [x for x in os.listdir(wd) if x.startswith("manager")]
    manager_pos_cnt = {}
    for m in managers:
        trials = os.listdir(os.path.join(wd, m, "weights"))
        pred = pd.read_table(os.path.join(wd, m, "weights", trials[0], "pred.txt"), comment="#")
        manager_pos_cnt[m] = pred['obs'].sum()
    return manager_pos_cnt


def plot_zs_hist(hist_fp, config_fp, save_prefix, zoom_first_n=None):
    zs_hist = pd.read_table(hist_fp, header=None, sep=",")
    configs = pd.read_table(config_fp)
    #zs_hist['task'] = zs_hist[0].apply(lambda x: configs.loc[int(x.split('-')[0])]['feat_name'])
    zs_hist['task'] = zs_hist[0].apply(lambda x: "Manager:%i"%int(x.split('-')[0]))
    zs_hist['task_int'] = zs_hist[0].apply(lambda x: int(x.split('-')[0]))
    zs_hist['trial'] = zs_hist[0].apply(lambda x: int(x.split('-')[1]))
    zs_hist['step'] = zs_hist['trial']//5
    zs_hist = zs_hist[[2,'task','step', 'task_int']].groupby(['task','step']).mean()
    zs_hist['auc'] = zs_hist[2]
    zs_hist = zs_hist.drop([2], axis=1)
    zs_hist['task'] = [a[0] for a in zs_hist.index]
    zs_hist['step'] = [a[1] for a in zs_hist.index]
    zs_hist['task_int'] = zs_hist['task_int']//10
    for i in zs_hist['task_int'].unique():
        zs_hist_ = zs_hist.loc[zs_hist.task_int==i]
        #zs_hist['auc'] = sma(zs_hist[2], window=20)
        #zs_hist['auc'] = zs_hist[2]
        #zs_hist = zs_hist.drop([0,1,2,3,4,5,6], axis=1)
        if zoom_first_n is not None:
            zs_hist_ = zs_hist_.loc[zs_hist_.trial <= zoom_first_n]
        fig, ax = plt.subplots(1,1, figsize=(10, 10))
        sns.lineplot(x="step", y="auc", hue="task", data=zs_hist_, ax=ax, marker="o")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig("%s.%i.png"%(save_prefix, i))
        plt.close()


def plot_single_run(feat_dirs, save_fp, zoom_first_n=None):
    dfs = []
    for d in feat_dirs:
        hist = pd.read_table(os.path.join(d, "train_history.csv"), header=None, sep=",")
        hist['task'] = os.path.basename(d)
        hist['trial'] = hist[0]
        hist['auc'] = sma(hist[2], window=20)
        hist = hist.drop([0,1,2,3,4,5,6], axis=1)
        if zoom_first_n is not None:
            hist = hist.loc[hist.trial <= zoom_first_n]
        dfs.append(hist)
    df = pd.concat(dfs)
    ax = sns.lineplot(x="trial", y="auc", hue="task", data=df)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(save_fp)
    plt.close()


#plot_zs_hist("./outputs/zs_50.2/train_history.csv", "data/zero_shot/train_feats.config_file.8_random_feats.tsv", "zs_hist.png", zoom_first_n=300)
plot_zs_hist("./outputs/new_20200915/ds_simple.ppo/train_history.csv", "data/zero_shot_deepsea/train_feats.config_file.tsv", "zs_simple.ppo")
plot_zs_hist("./outputs/new_20200915/ds_simple_space/train_history.csv", "data/zero_shot_deepsea/train_feats.config_file.tsv", "zs_simple.rl")

#feat_dirs = ["./outputs/%s"%x for x in os.listdir("./outputs/") if x.startswith("FEAT")]
#plot_single_run(feat_dirs, "single_runs.png", zoom_first_n=300)


