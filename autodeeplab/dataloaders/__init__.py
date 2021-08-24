import os
import numpy as np
import tarfile
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from utilities3 import *
from dataio import load_list
from generator import PDNetDataset
from cosmic_dataset import PairedDatasetImagePath, get_dirs

def make_data_loader(args, **kwargs):

    if args.dataset == 'darcyflow':
        TRAIN_PATH = 'piececonst_r421_N1024_smooth1.mat'
        TEST_PATH = 'piececonst_r421_N1024_smooth2.mat'
        ntrain = 900
        nval = 1000-ntrain
        if args.autodeeplab == 'train':
            ntrain += nval
            nval = 0
        ntest = 100
        r = 5
        h = int(((421 - 1)/r) + 1)
        s = h

        reader = MatReader(TRAIN_PATH)
        x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
        y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]
        if args.autodeeplab != 'train':
            x_val = reader.read_field('coeff')[ntrain:ntrain+nval,::r,::r][:,:s,:s]
            y_val = reader.read_field('sol')[ntrain:ntrain+nval,::r,::r][:,:s,:s]

        reader.load_file(TEST_PATH)
        x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
        y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]

        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        if args.autodeeplab != 'train':
            x_val = x_normalizer.encode(x_val)
        x_test = x_normalizer.encode(x_test)

        y_normalizer = UnitGaussianNormalizer(y_train)
        if args.autodeeplab != 'train':
            y_val = y_normalizer.encode(y_val)
        y_train = y_normalizer.encode(y_train)

        grids = []
        grids.append(np.linspace(0, 1, s))
        grids.append(np.linspace(0, 1, s))
        grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
        grid = grid.reshape(1,s,s,2)
        grid = torch.tensor(grid, dtype=torch.float)
        x_train = torch.cat([x_train.reshape(ntrain,s,s,1), grid.repeat(ntrain,1,1,1)], dim=3)
        if args.autodeeplab != 'train':
            x_val = torch.cat([x_val.reshape(nval,s,s,1), grid.repeat(nval,1,1,1)], dim=3)
        x_test = torch.cat([x_test.reshape(ntest,s,s,1), grid.repeat(ntest,1,1,1)], dim=3)

        train_loader = DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True, **kwargs)
        if args.autodeeplab != 'train':
            val_loader = DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False, **kwargs)
        if args.autodeeplab == 'train':
            return train_loader, y_normalizer
        return train_loader, train_loader, val_loader, test_loader, y_normalizer

    elif args.dataset == 'protein':
        if not os.path.isdir('deepcov'):
            import zipfile
            with zipfile.ZipFile('protein.zip', 'r') as zip_ref:
                zip_ref.extractall()
        training_window = 128
        expected_n_channels = 57
        pad_size = 10
        data_dir = '.'
        all_feat_paths = [data_dir + '/deepcov/features/', 
            data_dir + '/psicov/features/', data_dir + '/cameo/features/']
        all_dist_paths = [data_dir + '/deepcov/distance/', 
            data_dir + '/psicov/distance/', data_dir + '/cameo/distance/']
        deepcov_list = load_list(data_dir + '/deepcov.lst', -1)

        length_dict = {}
        for pdb in deepcov_list:
            (ly, seqy, cb_map) = np.load(
                data_dir + '/deepcov/distance/' + pdb + '-cb.npy', 
                allow_pickle = True)
            length_dict[pdb] = ly

        nval = 100
        if args.autodeeplab == 'train':
            nval = 0
        if args.autodeeplab != 'train':
            val_pdbs = deepcov_list[:nval]
        train_pdbs = deepcov_list[nval:]

        train_dataset = PDNetDataset(
            train_pdbs, all_feat_paths, all_dist_paths, training_window, pad_size, args.batch_size, expected_n_channels, label_engineering = '16.0')
        if args.autodeeplab != 'train':
            val_dataset = PDNetDataset(
                val_pdbs, all_feat_paths, all_dist_paths, training_window, pad_size, args.batch_size, expected_n_channels, label_engineering = '16.0')
        LMAX = 512
        psicov_list = load_list(data_dir + '/psicov.lst')
        psicov_length_dict = {}
        for pdb in psicov_list:
            (ly, seqy, cb_map) = np.load(data_dir + '/psicov/distance/' + pdb + '-cb.npy', allow_pickle=True)
            psicov_length_dict[pdb] = ly
        test_dataset = PDNetDataset(
            psicov_list, all_feat_paths, all_dist_paths, LMAX, pad_size, 1, expected_n_channels, label_engineering = None)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        if args.autodeeplab != 'train':
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=2, num_workers=0, pin_memory=True)

        if args.autodeeplab == 'train':
            return train_loader, 1
        return train_loader, train_loader, val_loader, test_loader, (psicov_list, psicov_length_dict)


    elif args.dataset == 'cosmic':
        base_dir = '.'
        os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
        data_base = os.path.join(base_dir, 'data')
        
        if not os.path.exists(os.path.join(base_dir, 'train_dirs.npy')):
            train_tar = tarfile.open(os.path.join(base_dir, 'deepCR.ACS-WFC.train.tar'))
            test_tar = tarfile.open(os.path.join(base_dir, 'deepCR.ACS-WFC.test.tar'))
            train_tar.extractall(data_base)
            test_tar.extractall(data_base)
            get_dirs(base_dir, data_base)
        train_dirs = np.load(os.path.join(base_dir, 'train_dirs.npy'), allow_pickle=True)
        test_dirs = np.load(os.path.join(base_dir, 'test_dirs.npy'), allow_pickle=True)
        print('got dirs')
        aug_sky = (-0.9, 3)

        # only train f435 and GAL flag for now
        if args.autodeeplab == 'train':
            train_data = PairedDatasetImagePath(train_dirs[::], aug_sky[0], aug_sky[1], part=None)
            train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, **kwargs)
            test_data = PairedDatasetImagePath(test_dirs[::], aug_sky[0], aug_sky[1], part=None)       
            test_loader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size, num_workers=0, pin_memory=True)
            data_shape = train_data[0][0].shape[1]

            return train_loader, test_loader, data_shape

        else:
            train_data = PairedDatasetImagePath(train_dirs[::], aug_sky[0], aug_sky[1], part='train')
            valid_data = PairedDatasetImagePath(train_dirs[::], aug_sky[0], aug_sky[1], part='test')
            test_data = PairedDatasetImagePath(test_dirs[::], aug_sky[0], aug_sky[1], part=None)
            print(len(train_data), len(valid_data), len(test_data))
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size, num_workers=0, pin_memory=True)
            data_shape = train_data[0][0].shape[1]

        return train_loader, train_loader, val_loader, test_loader, data_shape
            
    else:
        raise NotImplementedError
