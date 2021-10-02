'''
Author: Badri Adhikari, University of Missouri-St. Louis, 12-18-2019
File: Contains subroutines to visualize the inputs and predictions
'''

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import numpy as np

def plot_protein_io(X, Y):
    figure(num=None, figsize=(20, 54), dpi=80, facecolor='w', frameon=True, edgecolor='k')
    print('')
    print('Generating seaborn plots.. patience..')
    for i in range(0, len(X[0, 0, :])):
        plt.subplot(14, 4, i + 1)
        sns.heatmap(X[:, :, i], cmap='RdYlBu')
        plt.title('Channel ' + str(i))
    plt.subplot(14, 4, len(X[0, 0, :]) + 1)
    plt.grid(None)
    y = np.copy(Y)
    y[y > 25.0] = 25.0
    sns.heatmap(y, cmap='Spectral')
    plt.title('True Distances')
    plt.show()

def plot_learning_curves(history):
    print('')
    print('Curves..')
    print(history.params)
    plt.clf()
    if 'mean_absolute_error' in history.history:
        plt.plot(history.history['mean_absolute_error'], 'g', label = 'Training MAE')
        plt.plot(history.history['val_mean_absolute_error'], 'b', label = 'Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
    elif 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], 'g', label = 'Training Accuracy')
        plt.plot(history.history['val_accuracy'], 'b', label = 'Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
    else:
        plt.plot(history.history['mae'], 'g', label = 'Training MAE')
        plt.plot(history.history['val_mae'], 'b', label = 'Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
    plt.legend()
    plt.show()

def plot_four_pair_maps(T, P, pdb_list, length_dict):
    figure(num=None, figsize=(24, 10), dpi=60, facecolor='w', frameon=True, edgecolor='k')
    I = 1
    for k in range(4):
        L = length_dict[pdb_list[k]]
        plt.subplot(2, 4, I)
        I += 1
        sns.heatmap(T[k, 0:L, 0:L, 0], cmap='Spectral')
        plt.title('True - ' + pdb_list[k])
    for k in range(4):
        L = length_dict[pdb_list[k]]
        plt.subplot(2, 4, I)
        I += 1
        sns.heatmap(P[k, 0:L, 0:L, 0], cmap='Spectral')
        plt.title('Prediction - ' + pdb_list[k])
    plt.show()

def plot_channel_histograms(X):
    for i in range(len(x[0, 0, :])):
        print ('Input feature', i)
        plt.hist(x[:, :, i].flatten())
        plt.show()
    print('Output labels')
    plt.hist(y.flatten())
    plt.show()
