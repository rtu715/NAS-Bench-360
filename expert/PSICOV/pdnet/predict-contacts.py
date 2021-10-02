'''
Author: Badri Adhikari, University of Missouri-St. Louis,  12-29-2019
File: Contains the code to predict contacts
'''

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import sys
import numpy as np
import datetime
import pickle
import getopt

from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input, Convolution2D, Activation, add, Dropout, BatchNormalization
from tensorflow.python.keras.models import Model

if sys.version_info < (3,0,0):
    print('Python 3 required!!!')
    sys.exit(1)

# Some GPUs don't allow memory growth by default (keep both options)
# Option 1
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
# Option 2
#import keras.backend as K
#gpu_options = tf.GPUOptions(allow_growth=True)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#K.tensorflow_backend.set_session(sess)

def usage():
    print('Usage:')
    print(sys.argv[0] + ' <-w file_weights> <-p file_pkl> <-o outrr>')

try:
    opts, args = getopt.getopt(sys.argv[1:], "w:p:o:h")
except getopt.GetoptError as err:
    print(err)
    usage()
    sys.exit(2)

wts = ''
pkl = ''
rr  = ''
for o, a in opts:
    if o in ("-h", "--help"):
        usage()
        sys.exit()
    elif o in ("-w"):
        wts = os.path.abspath(a)
    elif o in ("-p"):
        pkl = os.path.abspath(a)
    elif o in ("-o"):
        rr = os.path.abspath(a)
    else:
        assert False, "Error!! unhandled option!!"

if len(wts) < 2:
    print('wts file undefined!')
    usage()
    sys.exit()
if len(pkl) < 2:
    print('in pkl undefined!')
    usage()
    sys.exit()
if len(rr) < 2:
    print('our rr undefined!')
    usage()
    sys.exit()

pad_size                  = 10
expected_n_channels       = 57
OUTL = 1024

def save_contacts_rr(seq, pred_matrix, file_rr):
    rr = open(file_rr, 'w')
    rr.write(seq + "\n")
    P = np.copy(pred_matrix)
    L = len(P[:])
    for j in range(0, L):
        for k in range(j, L):
            P[j, k] = (P[k, j, 0] + P[j, k, 0]) / 2.0
    for j in range(0, L):
        for k in range(j, L):
            if abs(j - k) < 5:
                continue
            rr.write("%i %i 0 8 %.5f\n" %(j+1, k+1, (P[j][k])) )
    rr.close()
    print('Written RR ' + file_rr + ' !')

def get_feature(infile, expected_n_channels):
    features = pickle.load(open(infile, 'rb'))
    l = len(features['seq'])
    seq = features['seq']
    # Create X and Y placeholders
    X = np.full((l, l, expected_n_channels), 0.0)
    # Add secondary structure
    ss = features['ss']
    assert ss.shape == (3, l)
    fi = 0
    for j in range(3):
        a = np.repeat(ss[j].reshape(1, l), l, axis = 0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    # Add PSSM
    pssm = features['pssm']
    assert pssm.shape == (l, 22)
    for j in range(22):
        a = np.repeat(pssm[:, j].reshape(1, l), l, axis = 0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    # Add SA
    sa = features['sa']
    assert sa.shape == (l, )
    a = np.repeat(sa.reshape(1, l), l, axis = 0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    # Add entrophy
    entropy = features['entropy']
    assert entropy.shape == (l, )
    a = np.repeat(entropy.reshape(1, l), l, axis = 0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    # Add CCMpred
    ccmpred = features['ccmpred']
    assert ccmpred.shape == ((l, l))
    X[:, :, fi] = ccmpred
    fi += 1
    # Add  FreeContact
    freecon = features['freecon']
    assert freecon.shape == ((l, l))
    X[:, :, fi] = freecon
    fi += 1
    # Add potential
    potential = features['potential']
    assert potential.shape == ((l, l))
    X[:, :, fi] = potential
    fi += 1
    assert fi == expected_n_channels
    assert X.max() < 100.0
    assert X.min() > -100.0
    return X

# Architecture DEEPCON (original)
def deepcon_rdd(L, num_blocks, width, expected_n_channels):
    print('')
    print('Model params:')
    print('L', L)
    print('num_blocks', num_blocks)
    print('width', width)
    print('expected_n_channels', expected_n_channels)
    print('')
    dropout_value = 0.3
    my_input = Input(shape = (L, L, expected_n_channels))
    tower = BatchNormalization()(my_input)
    tower = Activation('relu')(tower)
    tower = Convolution2D(width, 1, padding = 'same')(tower)
    n_channels = width
    d_rate = 1
    for i in range(num_blocks):
        block = BatchNormalization()(tower)
        block = Activation('relu')(block)
        block = Convolution2D(n_channels, kernel_size = (3, 3), padding = 'same')(block)
        block = Dropout(dropout_value)(block)
        block = Activation('relu')(block)
        block = Convolution2D(n_channels, kernel_size = (3, 3), dilation_rate=(d_rate, d_rate), padding = 'same')(block)
        tower = add([block, tower])
        if d_rate == 1:
            d_rate = 2
        elif d_rate == 2:
            d_rate = 4
        else:
            d_rate = 1
    tower = BatchNormalization()(tower)
    tower = Activation('relu')(tower)
    tower = Convolution2D(1, 3, padding = 'same')(tower)
    tower = Activation('sigmoid')(tower)
    model = Model(my_input, tower)
    return model

features = pickle.load(open(pkl, 'rb'))
l = len(features['seq'])
seq = features['seq']

OUTL = l + pad_size

X = get_feature(pkl, expected_n_channels)
assert len(X[0, 0, :]) == expected_n_channels
l = len(X[:, 0, 0])

XX = np.full((1, OUTL, OUTL, expected_n_channels), 0.0)
Xpadded = np.zeros((l + pad_size, l + pad_size, len(X[0, 0, :])))
Xpadded[int(pad_size/2) : l+int(pad_size/2), int(pad_size/2) : l+int(pad_size/2), :] = X
l = len(Xpadded[:, 0, 0])
XX[0, :l, :l, :] = Xpadded

print('')
print('Channel summaries:')
print(' Channel        Avg        Max        Sum')
for i in range(len(X[0, 0, :])):
    (m, s, a) = (X[:, :, i].flatten().max(), X[:, :, i].flatten().sum(), X[:, :, i].flatten().mean())
    print(' %7s %10.4f %10.4f %10.1f' % (i+1, a, m, s))

model = deepcon_rdd(OUTL, 128, 64, expected_n_channels)

model.load_weights(wts)

P = model.predict(XX)
# Remove padding, i.e. shift up and left by int(pad_size/2)
P[:, :OUTL-pad_size, :OUTL-pad_size, :] = P[:, int(pad_size/2) : OUTL-int(pad_size/2), int(pad_size/2) : OUTL-int(pad_size/2), :]

print('')
print('Save predictions..')
save_contacts_rr(seq, P[0, :len(seq), :len(seq)], rr)
