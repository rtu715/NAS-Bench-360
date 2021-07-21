'''
Credit: https://github.com/profjsb/deepCR
'''
import numpy as np
from torch.utils.data import Dataset
import os

def get_dirs(base_dir, data_base):
    train_dirs = []
    test_dirs = []

    test_base = os.path.join(data_base,'npy_test')
    train_base = os.path.join(data_base,'npy_train')

    print('------------------------------------------------------------')
    print('Fetching directories for the test set')
    print('------------------------------------------------------------')
    for _filter in os.listdir(test_base):
        filter_dir = os.path.join(test_base,_filter)
        if os.path.isdir(filter_dir):
            for prop_id in os.listdir(filter_dir):
                prop_id_dir = os.path.join(filter_dir,prop_id)
                if os.path.isdir(prop_id_dir):
                    for vis_num in os.listdir(prop_id_dir):
                        vis_num_dir = os.path.join(prop_id_dir,vis_num)
                        if os.path.isdir(vis_num_dir):
                            for f in os.listdir(vis_num_dir):
                                if '.npy' in f and f != 'sky.npy':
                                    test_dirs.append(os.path.join(vis_num_dir,f))

    print('------------------------------------------------------------')
    print('Fetching directories for the training set')
    print('------------------------------------------------------------')
    for _filter in os.listdir(train_base):
        filter_dir = os.path.join(train_base,_filter)
        if os.path.isdir(filter_dir):
            for prop_id in os.listdir(filter_dir):
                prop_id_dir = os.path.join(filter_dir,prop_id)
                if os.path.isdir(prop_id_dir):
                    for vis_num in os.listdir(prop_id_dir):
                        vis_num_dir = os.path.join(prop_id_dir,vis_num)
                        if os.path.isdir(vis_num_dir):
                            for f in os.listdir(vis_num_dir):
                                if '.npy' in f and f != 'sky.npy':
                                    train_dirs.append(os.path.join(vis_num_dir,f))
#     print(train_dirs)
    np.save(os.path.join(base_dir,'test_dirs.npy'), test_dirs)
    np.save(os.path.join(base_dir,'train_dirs.npy'), train_dirs)

    return None

class PairedDatasetImagePath(Dataset):
    def __init__(self, paths, skyaug_min=0, skyaug_max=0, part=None, f_val=0.1, seed=1):
        """ custom pytorch dataset class to load deepCR-mask training data
        :param paths: (list) list of file paths to (3, W, H) images: image, cr, ignore.
        :param skyaug_min: [float, float]. If sky is provided, use random sky background in the range
          [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers to allow the trained model to adapt to a
          wider range of sky background or equivalently exposure time. Remedy the fact that exposure time in the
          training set is discrete and limited.
        :param skyaug_min: float. subtract maximum amount of abs(skyaug_min) * sky_level as data augmentation
        :param skyaug_max: float. add maximum amount of skyaug_max * sky_level as data augmentation
        :param part: either 'train' or 'val'.
        :param f_val: percentage of dataset reserved as validation set.
        :param seed: fix numpy random seed to seed, for reproducibility.
        """
        assert 0 < f_val < 1
        np.random.seed(seed)
        n_total = len(paths)
        n_train = int(n_total * (1 - f_val)) #int(len * (1 - f_val)) JK
        f_test = f_val
        n_search = int(n_total * (1 - f_val - f_test))

        if part == 'train':
            s = np.s_[:max(1, n_train)]
        elif part == 'test':
            s = np.s_[min(n_total - 1, n_train):]

        elif part == 'search':
            s = np.s_[:max(1, n_search)]
        elif part == 'val':
            s = np.s_[min(n_total-1, n_search), n_train]

        else:
            s = np.s_[0:]

        self.paths = paths[s]
        self.skyaug_min = skyaug_min
        self.skyaug_max = skyaug_max

    def __len__(self):
        return len(self.paths)

    def get_skyaug(self, i):
        """
        Return the amount of background flux to be added to image
        The original sky background should be saved in sky.npy in each sub-directory
        Otherwise always return 0
        :param i: index of file
        :return: amount of flux to add to image
        """
        path = os.path.split(self.paths[i])[0]
        sky_path = os.path.join(path, 'sky.npy') #JK
        if os.path.isfile(sky_path):
            f_img = self.paths[i].split('/')[-1]
            sky_idx = int(f_img.split('_')[0])
            sky = np.load(sky_path)[sky_idx-1]
            return sky * np.random.uniform(self.skyaug_min, self.skyaug_max)
        else:
            return 0

    def __getitem__(self, i):
        data = np.load(self.paths[i])
        image = data[0]
        mask = data[1]
        if data.shape[0] == 3:
            ignore = data[2]
        else:
            ignore = np.zeros_like(data[0])
        # try:#JK
        skyaug = self.get_skyaug(i)
        return image + skyaug, mask, ignore
