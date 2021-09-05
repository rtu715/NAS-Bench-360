import numpy as np
from torch.utils.data import Dataset


class BilevelDataset(Dataset):
    def __init__(
        self,
        dataset,
    ):
        """
        We will split the data into a train split and a validation split
        and return one image from each split as a single observation.
        Args:
            dataset: PyTorch Dataset object
        """
        inds = np.arange(len(dataset))
        self.dataset = dataset
        # Make sure train and val splits are of equal size.
        # This is so we make sure to loop images in both train
        # and val splits exactly once in an epoch.
        n_train = int(0.2 * len(inds))
        self.train_inds1 = inds[0:n_train]
        self.train_inds2 = inds[n_train: 2*n_train]
        self.train_inds3 = inds[2*n_train: 3*n_train]
        self.train_inds4 = inds[3*n_train: 4*n_train]
        self.val_inds = inds[4*n_train : 5*n_train]
        assert len(self.train_inds1) == len(self.val_inds)

    def shuffle_val_inds(self):
        # This is so we will see different pairs of images
        # from train and val splits.  Will need to call this
        # manually at epoch end.
        np.random.shuffle(self.val_inds)

    def __len__(self):
        return len(self.train_inds1)

    def __getitem__(self, idx):
        train_ind1 = self.train_inds1[idx]
        train_ind2 = self.train_inds2[idx]
        train_ind3 = self.train_inds3[idx]
        train_ind4 = self.train_inds4[idx]
        val_ind = self.val_inds[idx]
        x_train1, y_train1 = self.dataset[train_ind1]
        x_train2, y_train2 = self.dataset[train_ind2]
        x_train3, y_train3 = self.dataset[train_ind3]
        x_train4, y_train4 = self.dataset[train_ind4]
        x_val, y_val = self.dataset[val_ind]
        return x_train1, y_train1, x_train2, y_train2, x_train3, y_train3, x_train4, y_train4, x_val, y_val

class BilevelCosmicDataset(Dataset):
    def __init__(
        self,
        dataset,
    ):
        """
        We will split the data into a train split and a validation split
        and return one image from each split as a single observation.
        Args:
            dataset: PyTorch Dataset object
        """
        inds = np.arange(len(dataset))
        self.dataset = dataset
        # Make sure train and val splits are of equal size.
        # This is so we make sure to loop images in both train
        # and val splits exactly once in an epoch.
        n_train = int(0.2 * len(inds))
        self.train_inds1 = inds[0:n_train]
        self.train_inds2 = inds[n_train: 2*n_train]
        self.train_inds3 = inds[2*n_train: 3*n_train]
        self.train_inds4 = inds[3*n_train: 4*n_train]
        self.val_inds = inds[4*n_train : 5*n_train]
        assert len(self.train_inds1) == len(self.val_inds)

    def shuffle_val_inds(self):
        # This is so we will see different pairs of images
        # from train and val splits.  Will need to call this
        # manually at epoch end.
        np.random.shuffle(self.val_inds)

    def __len__(self):
        return len(self.train_inds1)

    def __getitem__(self, idx):
        train_ind1 = self.train_inds1[idx]
        train_ind2 = self.train_inds2[idx]
        train_ind3 = self.train_inds3[idx]
        train_ind4 = self.train_inds4[idx]
        val_ind = self.val_inds[idx]
        img1, mask1, ignore1 = self.dataset[train_ind1]
        img2, mask2, ignore2 = self.dataset[train_ind2]
        img3, mask3, ignore3 = self.dataset[train_ind3]
        img4, mask4, ignore4 = self.dataset[train_ind4]
        img_val, mask_val, ignore_val = self.dataset[val_ind]
        return img1, mask1, ignore1, img2, mask2, ignore2, img3, mask3, ignore3, \
               img4, mask4, ignore4, img_val, mask_val, ignore_val

