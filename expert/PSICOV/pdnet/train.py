'''
Author: Badri Adhikari, University of Missouri-St. Louis,  11-15-2020
File: Contains the code to train and test learning real-valued distances, binned-distances and contact maps
'''

import os
import sys
import numpy as np
import datetime
import argparse

import keras.backend as K
def get_flops():
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
				run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops #Prints the "flops" of the model.

flag_plots = False

if flag_plots:
    #%matplotlib inline
    from plots import *

if sys.version_info < (3,0,0):
    print('Python 3 required!!!')
    sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            epilog='EXAMPLE:\npython3 train.py -w distance.hdf5 -n -1 -c 128 -e 64 -d 128 -f 16 -p ./ -v 0 -o tmp/')
    parser.add_argument('-w', type=str, required = True, dest = 'file_weights', help="hdf5 weights file")
    parser.add_argument('-n', type=int, required = True, dest = 'dev_size', help="number of pdbs to use for training (use -1 for ALL)")
    parser.add_argument('-c', type=int, required = True, dest = 'training_window', help="crop size (window) for training, 64, 128, etc. ")
    parser.add_argument('-e', type=int, required = True, dest = 'training_epochs', help="# of epochs")
    parser.add_argument('-o', type=str, required = True, dest = 'dir_out', help="directory to write .npy files")
    parser.add_argument('-d', type=int, required = True, dest = 'arch_depth', help="residual arch depth")
    parser.add_argument('-f', type=int, required = True, dest = 'filters_per_layer', help="number of convolutional filters in each layer")
    parser.add_argument('-p', type=str, required = True, dest = 'dir_dataset', help="path where all the data (including .lst) is located")
    parser.add_argument('-v', type=int, required = True, dest = 'flag_eval_only', help="1 = Evaluate only, don't train")
    args = parser.parse_args()
    return args

args = get_args()

file_weights              = args.file_weights
dev_size                  = args.dev_size
training_window           = args.training_window
training_epochs           = args.training_epochs
arch_depth                = args.arch_depth
filters_per_layer         = args.filters_per_layer
dir_dataset               = args.dir_dataset
dir_out                   = args.dir_out
flag_eval_only            = False
if args.flag_eval_only == 1:
    flag_eval_only = True
pad_size                  = 10
batch_size                = 2
expected_n_channels       = 57

# Import after argparse because this can throw warnings with "-h" option
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from dataio import *
from metrics import *
from generator import *
from models import *
from losses import *
from calculate_mae import *

np.random.seed(2)
tf.random.set_random_seed(2)

# Allow GPU memory growth
if hasattr(tf, 'GPUOptions'):
    import keras.backend as K
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.tensorflow_backend.set_session(sess)
else:
    # For other GPUs
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

print('Start ' + str(datetime.datetime.now()))

print('')
print('Parameters:')
print('dev_size', dev_size)
print('file_weights', file_weights)
print('training_window', training_window)
print('training_epochs', training_epochs)
print('arch_depth', arch_depth)
print('filters_per_layer', filters_per_layer)
print('pad_size', pad_size)
print('batch_size', batch_size)
print('dir_dataset', dir_dataset)
print('dir_out', dir_out)

os.system('mkdir -p ' + dir_out)

all_feat_paths = [dir_dataset + '/deepcov/features/', dir_dataset + '/psicov/features/', dir_dataset + '/cameo/features/']
all_dist_paths = [dir_dataset + '/deepcov/distance/', dir_dataset + '/psicov/distance/', dir_dataset + '/cameo/distance/']

deepcov_list = load_list(dir_dataset + '/deepcov.lst', dev_size)

length_dict = {}
for pdb in deepcov_list:
    (ly, seqy, cb_map) = np.load(dir_dataset + '/deepcov/distance/' + pdb + '-cb.npy', allow_pickle = True)
    length_dict[pdb] = ly

print('')
print('Split into training and validation set..')
#valid_pdbs = deepcov_list[:int(0.3 * len(deepcov_list))]
#train_pdbs = deepcov_list[int(0.3 * len(deepcov_list)):]
#if len(deepcov_list) > 200:
#    valid_pdbs = deepcov_list[:100]
#    train_pdbs = deepcov_list[100:]
train_pdbs = deepcov_list[:]

#print('Total validation proteins : ', len(valid_pdbs))
print('Total training proteins   : ', len(train_pdbs))

print('')
#print('Validation proteins: ', valid_pdbs)

train_generator = DistGenerator(train_pdbs, all_feat_paths, all_dist_paths, training_window, pad_size, batch_size, expected_n_channels, label_engineering = '16.0')
#valid_generator = DistGenerator(valid_pdbs, all_feat_paths, all_dist_paths, training_window, pad_size, batch_size, expected_n_channels, label_engineering = '16.0')

print('')
print('len(train_generator) : ' + str(len(train_generator)))
#print('len(valid_generator) : ' + str(len(valid_generator)))

X, Y = train_generator[1]
print('Actual shape of X    : ' + str(X.shape))
print('Actual shape of Y    : ' + str(Y.shape))

print('')
print('Channel summaries:')
summarize_channels(X[0, :, :, :], Y[0])

if flag_plots:
    print('')
    print('Inputs/Output of protein', 0)
    plot_protein_io(X[0, :, :, :], Y[0, :, :, 0])

print('')
print('Build a model..')
model = ''
model = deepcon_rdd_distances(training_window, arch_depth, filters_per_layer, expected_n_channels)

print('')
print('Compile model..')
model.compile(loss = 'logcosh', optimizer = 'rmsprop', metrics = ['mae'])

print(model.summary())
print(get_flops())

if flag_eval_only == 0:
    if os.path.exists(file_weights):
        print('')
        print('Loading existing weights..')
        model.load_weights(file_weights)
    print('')
    print('Train..')
    history = model.fit_generator(generator = train_generator,
        #validation_data = valid_generator,
        callbacks = [ModelCheckpoint(filepath = file_weights, save_weights_only = True, verbose = 1)],
        verbose = 1,
        max_queue_size = 8,
        workers = 1,
        use_multiprocessing = False,
        shuffle = True ,
        epochs = training_epochs)

    if flag_plots:
        plot_learning_curves(history)

psicov_list = load_list(dir_dataset + 'psicov.lst')
psicov_length_dict = {}
for pdb in psicov_list:
    (ly, seqy, cb_map) = np.load(dir_dataset + '/psicov/distance/' + pdb + '-cb.npy', allow_pickle = True)
    psicov_length_dict[pdb] = ly

'''
cameo_list = load_list(dir_dataset + 'cameo-hard.lst')
cameo_length_dict = {}
for pdb in cameo_list:
    (ly, seqy, cb_map) = np.load(dir_dataset + '/cameo/distance/' + pdb + '-cb.npy', allow_pickle = True)
    cameo_length_dict[pdb] = ly
'''
evalsets = {}
#evalsets['validation'] = {'LMAX': 512,  'list': valid_pdbs, 'lendict': length_dict}
evalsets['psicov'] = {'LMAX': 512,  'list': psicov_list, 'lendict': psicov_length_dict}
#evalsets['cameo']  = {'LMAX': 1300, 'list': cameo_list,  'lendict': cameo_length_dict}

for my_eval_set in evalsets:
    print('')
    print(f'Evaluate on the {my_eval_set} set..')
    my_list = evalsets[my_eval_set]['list']
    LMAX = evalsets[my_eval_set]['LMAX']
    length_dict = evalsets[my_eval_set]['lendict']
    print('L', len(my_list))
    print(my_list)

    model = deepcon_rdd_distances(LMAX, arch_depth, filters_per_layer, expected_n_channels)
    model.load_weights(file_weights)
    my_generator = DistGenerator(my_list, all_feat_paths, all_dist_paths, 512, 10, 1, 57, label_engineering = None)
    print(my_generator[0][0].shape)
    # Padded but full inputs/outputs
    P = model.predict_generator(my_generator, max_queue_size=10, verbose=1)
    Y = np.full((len(my_generator), LMAX, LMAX, 1), np.nan)
    for i, xy in enumerate(my_generator):
        Y[i, :, :, 0] = xy[1][0, :, :, 0]
    # Average the predictions from both triangles
    for j in range(0, len(P[0, :, 0, 0])):
        for k in range(j, len(P[0, :, 0, 0])):
            P[ :, j, k, :] = (P[ :, k, j, :] + P[ :, j, k, :]) / 2.0
    P[ P < 0.01 ] = 0.01
    # Remove padding, i.e. shift up and left by int(pad_size/2)
    P[:, :LMAX-pad_size, :LMAX-pad_size, :] = P[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]
    Y[:, :LMAX-pad_size, :LMAX-pad_size, :] = Y[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]
    # Recover the distance translations
    #P = 100.0 / (P + epsilon)
    '''
    print('')
    print('Evaluating distances..')
    results_list = evaluate_distances(P, Y, my_list, length_dict)
    print('')
    numcols = len(results_list[0].split())
    print(f'Averages for {my_eval_set}', end = ' ')
    for i in range(2, numcols):
        x = results_list[0].split()[i].strip()
        if x == 'count' or results_list[0].split()[i-1].strip() == 'count':
            continue
        avg_this_col = False
        if x == 'nan':
            avg_this_col = True
        try:
            float(x)
            avg_this_col = True
        except ValueError:
            None
        if not avg_this_col:
            print(x, end=' ')
            continue
        avg = 0.0
        count = 0
        for mrow in results_list:
            a = mrow.split()
            if len(a) != numcols:
                continue
            x = a[i]
            if x == 'nan':
                continue
            try:
                avg += float(x)
                count += 1
            except ValueError:
                print(f'ERROR!! float value expected!! {x}')
        print(f'AVG: {avg/count:.4f} items={count}', end = ' ')
    print('')
    
    if flag_plots:
        plot_four_pair_maps(Y, P, my_list, my_length_dict)
    '''
    print(calculate_mae(P, Y, my_list, length_dict))
    print('')
    print('')
    print('Save predictions..')
    for i in range(len(my_list)):
        L = length_dict[my_list[i]]
        np.save(dir_out + '/' + my_list[i] + '.npy', P[i, :L, :L, 0])

print('')
print ('Everything done! ' + str(datetime.datetime.now()) )
