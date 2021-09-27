'''
Author: Badri Adhikari, University of Missouri-St. Louis, 1-26-2020
File: Contains the metrics to evaluate predicted real-valued distances, binned-distances and contact maps
'''

import numpy as np
import tensorflow as tf
epsilon = tf.keras.backend.epsilon()

from dataio import *
from plots import *
from generator import *
from disteval import *

def evaluate_distances(PRED, YTRUE, pdb_list, length_dict):
    results_list = []
    for i in range(0, len(PRED[:, 0, 0, 0])):
        L = length_dict[pdb_list[i]]
        D = PRED[i, :L, :L, 0]
        ND = np.copy(YTRUE[i, 0:L, 0:L, 0])
        eval_dict = {}
        # Obtain precision values
        C = 4.0 / (D + 0.000001)
        C [C > 1.0] = 1.0
        NC = np.copy(ND)
        NC[NC < 8.0] = 1
        NC[NC >= 8.0] = 0
        eval_dict = calc_contact_errors_various_xl(CPRED = C, CTRUE = NC, separation = [12, 24])
        # Distance metrics
        eval_dict.update(calc_dist_errors_various_xl(P = D, Y = ND, L = L))
        # Obtain Cb-LDDT scores
        for S in [6, 12, 24]:
            for R in [15]:
                eval_dict[f"Cb-LDDT Radius: {R:2d} min-seq-sep: {S:2d}"] = get_LDDT(ND, D, R, S)
        for item in eval_dict:
            print(pdb_list[i], i, len(pdb_list), item, eval_dict[item])
        output = str(pdb_list[i]) + ' ' + str(L) + ' ' + str(i) + ' ' + str(eval_dict)
        output = output.replace(',', '')
        output = output.replace('{', '')
        output = output.replace('}', '')
        output = output.replace(':', '')
        output = output.replace('\'', '')
        results_list.append(output)
    return results_list

def eval_contact_predictions(my_model, my_list, my_length_dict, my_dir_features, my_dir_distance, pad_size, flag_plots, flag_save, LMAX, expected_n_channels):
    # Padded but full inputs/outputs
    my_generator = ContactGenerator(my_list, my_dir_features, my_dir_distance, LMAX, pad_size, 1, expected_n_channels)
    P = my_model.predict_generator(my_generator, max_queue_size=10, verbose=1)
    # Remove padding, i.e. shift up and left by int(pad_size/2)
    P[:, :LMAX-pad_size, :LMAX-pad_size, :] = P[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]
    Y = get_bulk_output_contact_maps(my_list, my_dir_distance, LMAX)

    if flag_plots:
        plot_four_pair_maps(Y, P, my_list, my_length_dict)

    print('')
    calculate_contact_precision(P, Y, my_list, my_length_dict)

    if flag_save:
        os.system('mkdir -p ./predictions')
        print('')
        print('Save predictions..')
        for i in range(len(my_list)):
            L = my_length_dict[my_list[i]]
            pred = P[i, :L, :L]
            save_contacts_rr(my_list[i], my_dir_features, pred, './predictions/' + my_list[i] + '.contacts.rr')

def eval_binned_predictions(my_model, my_list, my_length_dict, my_dir_features, my_dir_distance, pad_size, flag_plots, flag_save, LMAX, bins, expected_n_channels):
    # Padded but full inputs/outputs
    my_generator = BinnedDistGenerator(my_list, my_dir_features, my_dir_distance, bins, LMAX, pad_size, 1, expected_n_channels)
    P = my_model.predict_generator(my_generator, max_queue_size=10, verbose=1)
    # Remove padding, i.e. shift up and left by int(pad_size/2)
    P[:, :LMAX-pad_size, :LMAX-pad_size, :] = P[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]
    Y = get_bulk_output_contact_maps(my_list, my_dir_distance, LMAX)

    # Predicted distance is mean distance of the most confident bin
    D = np.zeros((len(P), LMAX, LMAX, 1))
    for p in range(len(P)):
        for i in range(LMAX):
            for j in range(LMAX):
                index = np.argmax(P[p, i, j, :])
                min_max = [float(x) for x in bins[index].split()]
                D[p, i, j, 0] = ( min_max[0] + min_max[1] ) / 2.0

    # The last bin's range has a very large value, so trim it
    bin_max = float(bins[len(bins) - 1].split()[0])
    D[D > bin_max] = bin_max

    Y = get_bulk_output_dist_maps(my_list, my_dir_distance, LMAX)

    print('')
    calculate_mae(D, Y, my_list, my_length_dict)

    # Identify the bins that fall under the 8.0A distance
    contact_bins = -1
    for k, v in bins.items():
        if bins[k].split()[0] == '8.0':
            contact_bins = k

    # Sum the probabilities of the bins that fall under 8.0A distance
    C = np.zeros((len(P), LMAX, LMAX, 1))
    for p in range(len(P)):
        for i in range(LMAX):
            for j in range(LMAX):
                C[p, i, j, 0] = np.sum(P[p, i, j, :contact_bins])

    Y = get_bulk_output_contact_maps(my_list, my_dir_distance, LMAX)

    print('')
    calculate_contact_precision(C, Y, my_list, my_length_dict)

    if flag_save:
        os.system('mkdir -p ./predictions')
        print('')
        print('Save predictions..')
        for i in range(len(my_list)):
            L = my_length_dict[my_list[i]]
            predictions = {}
            for b in range(len(bins)):
                predictions[bins[b]] = P[i, :L, :L, b].astype(np.float16)
            f = open('./predictions/' + my_list[i] + '.bins.pkl', 'wb')
            pickle.dump(predictions, f)
            f.close()
