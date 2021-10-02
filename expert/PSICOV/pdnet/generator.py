'''
Author: Badri Adhikari, University of Missouri-St. Louis, 12-18-2019
File: Contains the code to "generate" protein I/O using data generators
'''

from tensorflow.keras.utils import *
from dataio import *
import tensorflow as tf
epsilon = tf.keras.backend.epsilon()

class ContactGenerator(Sequence):
    def __init__(self, pdb_id_list, features_path, distmap_path, dim, pad_size, batch_size, expected_n_channels):
        self.pdb_id_list = pdb_id_list
        self.features_path = features_path
        self.dim = dim
        self.pad_size = pad_size
        self.distmap_path = distmap_path
        self.batch_size = batch_size
        self.expected_n_channels = expected_n_channels

    def on_epoch_begin(self):
        self.indexes = np.arange(len(self.pdb_id_list))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return int(len(self.pdb_id_list) / self.batch_size)

    def __getitem__(self, index):
        batch_list = self.pdb_id_list[index * self.batch_size: (index + 1) * self.batch_size]
        X, Y = get_input_output_dist(batch_list, self.features_path, self.distmap_path, self.pad_size, self.dim, self.expected_n_channels)
        Y[ Y < 8.0  ] = 1.0
        Y[ Y >= 8.0 ] = 0.0
        return X, Y

class BinnedDistGenerator(Sequence):
    def __init__(self, pdb_id_list, features_path, distmap_path, bins, dim, pad_size, batch_size, expected_n_channels):
        self.pdb_id_list = pdb_id_list
        self.features_path = features_path
        self.dim = dim
        self.pad_size = pad_size
        self.distmap_path = distmap_path
        self.bins = bins
        self.batch_size = batch_size
        self.expected_n_channels = expected_n_channels

    def on_epoch_begin(self):
        self.indexes = np.arange(len(self.pdb_id_list))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return int(len(self.pdb_id_list) / self.batch_size)

    def __getitem__(self, index):
        batch_list = self.pdb_id_list[index * self.batch_size: (index + 1) * self.batch_size]
        X, Y = get_input_output_bins(batch_list, self.features_path, self.distmap_path, self.pad_size, self.dim, self.expected_n_channels, self.bins)
        return X, Y

class DistGenerator(Sequence):
    def __init__(self, pdb_id_list, features_path, distmap_path, dim, pad_size, batch_size, expected_n_channels, label_engineering = None):
        self.pdb_id_list = pdb_id_list
        self.features_path = features_path
        self.distmap_path = distmap_path
        self.dim = dim
        self.pad_size = pad_size
        self.batch_size = batch_size
        self.expected_n_channels = expected_n_channels
        self.label_engineering = label_engineering

    def on_epoch_begin(self):
        self.indexes = np.arange(len(self.pdb_id_list))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return int(len(self.pdb_id_list) / self.batch_size)

    def __getitem__(self, index):
        batch_list = self.pdb_id_list[index * self.batch_size: (index + 1) * self.batch_size]
        X, Y = get_input_output_dist(batch_list, self.features_path, self.distmap_path, self.pad_size, self.dim, self.expected_n_channels)
        if self.label_engineering is None:
            return X, Y
        if self.label_engineering == '100/d':
            return X, 100.0 / Y
        try:
            t = float(self.label_engineering)
            Y[Y > t] = t
        except ValueError:
            print('ERROR!! Unknown label_engineering parameter!!')
            return 
        return X, Y

