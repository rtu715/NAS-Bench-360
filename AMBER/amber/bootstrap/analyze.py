import matplotlib
import numpy as np
import pandas as pd
from ..utils.io import read_history

import matplotlib.pyplot as plt
import os
import json

matplotlib.use('Agg')
pd.set_option('display.expand_frame_repr', False)
working_dir = './tmp_mock/'


def get_rating(df, tmp, metrics_dict):
    idx_bool = np.array([df[i] == v for i, v in metrics_dict.items()])
    index = np.apply_along_axis(func1d=lambda x: all(x), axis=0, arr=idx_bool)
    r = np.random.choice(np.where(index)[0]).item()
    rank = tmp.index[tmp.id == df.iloc[r]['ID']][0]

    return rank


def get_config(df, tmp, metrics_dict):
    idx_bool = np.array([df[i] == v for i, v in metrics_dict.items()])
    index = np.apply_along_axis(func1d=lambda x: all(x), axis=0, arr=idx_bool)
    r = np.random.choice(np.where(index)[0]).item()
    model_state_str = [df.iloc[r]['ID'], df.iloc[r]['L1'], df.iloc[r]['L2'], df.iloc[r]['L3'], df.iloc[r]['L4']]

    return model_state_str


def main():
    hist_file_list = ["BioNAS/resources/mock_black_box/tmp_%i/train_history.csv" % i for i in range(1, 21)]

    data = json.load(open('./tmp_mock/metrics_vs_lambda.json'))
    df = read_history(hist_file_list)

    ktmp = pd.DataFrame([[i + 1, df.loc[i]['knowledge'].median()] for i in range(216)], columns=['id', 'value'])
    ktmp.sort_values(by=['value'], inplace=True)
    ktmp.index = range(len(ktmp))

    ltmp = pd.DataFrame([[i + 1, df.loc[i]['loss'].median()] for i in range(216)], columns=['id', 'value'])
    ltmp.sort_values(by=['value'], inplace=True)
    ltmp.index = range(len(ltmp))

    L = []

    lambda_list = [0.01, 0.1, 1.0, 10., 100.]

    for i, (a, l, k) in enumerate(zip(data['acc'], data['loss'], data['knowledge'])):
        for j, (a_, l_, k_) in enumerate(zip(a, l, k)):
            # break
            metric_and_loss = {'acc': a_, 'loss': l_, 'knowledge': k_}
            k_rank = get_rating(df, ktmp, metric_and_loss)
            l_rank = get_rating(df, ltmp, metric_and_loss)

            model_state_str = get_config(df, ltmp, metric_and_loss)

            L.append(model_state_str + [a_, l_, k_, lambda_list[j], k_rank, l_rank])

            data['k_rank'][i][j] = k_rank
            data['l_rank'][i][j] = l_rank

    output = pd.DataFrame(L,
                          columns=['id', 'L1', 'L2', 'L3', 'L4', 'acc', 'loss', 'knowledge', 'lambda', 'knowledge rank',
                                   'loss rank'])
    output.to_csv('./tmp_mock/best_configs.csv')
    plot(data, lambda_list)


def print_gold_standard():
    hist_file_list = ["BioNAS/resources/mock_black_box/tmp_%i/train_history.csv" % i for i in range(1, 21)]

    df = read_history(hist_file_list)

    ktmp = pd.DataFrame([[i, df.loc[i]['knowledge'].mean()] for i in range(216)], columns=['id', 'value'])
    ktmp.sort_values(by=['value'], inplace=True)
    ktmp.index = range(len(ktmp))

    print('Knowledge Gold Standard:', ktmp.iloc[0]['value'])
    print(df.loc[ktmp.iloc[0]['id']].iloc[0])

    ltmp = pd.DataFrame([[i, df.loc[i]['loss'].mean()] for i in range(216)], columns=['id', 'value'])
    ltmp.sort_values(by=['value'], inplace=True)
    ltmp.index = range(len(ltmp))

    print()
    print('Loss Gold Standard:', ltmp.iloc[0]['value'])
    print(df.loc[ltmp.iloc[0]['id']].iloc[0])


def plot(df, lambda_list):
    plt.close()
    for key in df:
        if key != 'k_rank' and key != 'l_rank' and key != 'acc':
            d = np.array(df[key])
            d = np.sort(d, axis=0)
            min_, max_, mid_ = d[int(d.shape[0] * 0.1), :], d[int(d.shape[0] * 0.9), :], d[int(d.shape[0] * 0.5), :]
            plt.plot(lambda_list, mid_, label=key, marker='o')
            plt.fill_between(lambda_list, min_, max_, alpha=0.2)

    plt.legend(loc='best')
    plt.xscale('log', basex=10)
    plt.xlabel('Lambda Value', fontsize=16)
    plt.ylabel('Metrics', fontsize=16)
    plt.title('Metrics of Best Config Found vs. Knowledge Weight')
    plt.savefig(os.path.join(working_dir, 'best_metrics_vs_lambda.pdf'))

    ## plot rank
    plt.close()
    for key in ['k_rank', 'l_rank']:
        d = np.array(df[key])
        d = np.sort(d, axis=0)
        min_, max_, mid_ = d[int(d.shape[0] * 0.1), :], d[int(d.shape[0] * 0.9), :], d[int(d.shape[0] * 0.5), :]
        if key == 'k_rank':
            label = 'knowledge rank'
        else:
            label = 'loss rank'
        plt.plot(lambda_list, mid_, label=label, marker='o')
        plt.fill_between(lambda_list, min_, max_, alpha=0.2)
    plt.legend(loc='best')
    plt.xscale('log', basex=10)
    plt.xlabel('Lambda Value', fontsize=16)
    plt.ylabel('Rank', fontsize=16)
    plt.title('Rank of Best Config Found vs. Knowledge Weight')
    plt.savefig(os.path.join(working_dir, 'best_rank_vs_lambda.pdf'))


if __name__ == '__main__1':
    print_gold_standard()
