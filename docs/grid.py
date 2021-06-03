import os
import torch
import numpy as np

from utils_grid import MatReader, UnitGaussianNormalizer
from utils_grid import create_grid
from protein_io import load_list
from protein_gen import PDNetDataset


def load_pde_data(path, sub, train=True):
    TRAIN_PATH = os.path.join(path, 'piececonst_r421_N1024_smooth1.mat')
    reader = MatReader(TRAIN_PATH)
    grid, s = create_grid(sub)
    r = sub
    ntrain = 1000
    ntest = 100
    if train:
        x_train = reader.read_field('coeff')[:ntrain - ntest, ::r, ::r][:, :s, :s]
        y_train = reader.read_field('sol')[:ntrain - ntest, ::r, ::r][:, :s, :s]

        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)

        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)

        ntrain = ntrain - ntest
        x_train = torch.cat([x_train.reshape(ntrain, s, s, 1), grid.repeat(ntrain, 1, 1, 1)], dim=3)
        train_data = torch.utils.data.TensorDataset(x_train, y_train)

    else:
        x_train = reader.read_field('coeff')[:ntrain, ::r, ::r][:, :s, :s]
        y_train = reader.read_field('sol')[:ntrain, ::r, ::r][:, :s, :s]

        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)

        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = y_normalizer.encode(y_train)

        x_train = torch.cat([x_train.reshape(ntrain, s, s, 1), grid.repeat(ntrain, 1, 1, 1)], dim=3)
        train_data = torch.utils.data.TensorDataset(x_train, y_train)

    return train_data

def load_protein_data(path, train=True):
    os.chdir(path)
    import zipfile
    with zipfile.ZipFile('protein.zip', 'r') as zip_ref:
        zip_ref.extractall()
    all_feat_paths = [path + '/deepcov/features/',
                           path + '/psicov/features/', path + '/cameo/features/']
    all_dist_paths =  [path + '/deepcov/distance/',
                         path + '/psicov/distance/', path + '/cameo/distance/']

    deepcov_list = load_list('deepcov.lst', -1)

    length_dict = {}
    for pdb in deepcov_list:
        (ly, seqy, cb_map) = np.load(
            'deepcov/distance/' + pdb + '-cb.npy',
            allow_pickle=True)
        length_dict[pdb] = ly

    if train:
        train_pdbs = deepcov_list[100:]

        train_data = PDNetDataset(train_pdbs,all_feat_paths,all_dist_paths,
                                  128, 10, 8, 57,
                                  label_engineering='16.0')

    else:
        train_pdbs =deepcov_list[:]
        train_data = PDNetDataset(train_pdbs, all_feat_paths, all_dist_paths,
                                  128, 10, 8 , 57,
                                  label_engineering='16.0')

    return train_data