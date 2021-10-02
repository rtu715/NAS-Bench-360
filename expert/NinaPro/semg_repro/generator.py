import numpy as np
from tensorflow.keras.utils import to_categorical, Sequence

from data import dataset, ma_batch, add_noise_random

class generator(Sequence):
    def __init__(
        self,
        d: dataset,
        repetitions: list,
        shuffle: bool = True,
        batch_size: int = 128,
        imu: bool = False,
        augment: bool = True,
        ma: bool = True,
        has_zero: bool = True
    ) -> None:
        self.shuffle = shuffle
        self.augment = augment
        self.ma = ma
        self.ma_len = d.ma
        self.repetitions = repetitions
        self.batch_size = batch_size
        self.imu = imu
        self.has_zero = has_zero
        idxs = np.where(np.isin(d.repetition, np.array(repetitions)))
        self.X = d.emg[idxs].copy()
        if imu:
            self.X = np.concatenate([self.X.copy(), d.imu[idxs].copy()], axis=-1)
        self.y = to_categorical(d.labels[idxs].copy())
        self.on_epoch_end()

    def __len__(self):
        "number of batches per epoch"
        return int(np.floor(self.X.shape[0] / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(self.X.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'generate a single batch'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        out = self.X[indexes,:,:].copy()
        if self.augment:
            for i in range(out.shape[0]):
                if self.has_zero:
                    if self.y[i,0] == 1:
                        out[i,:,:] = out[i,:,:]
                    else:
                        out[i,:,:]=add_noise_random(out[i,:,:])
                else:
                    out[i,:,:] = add_noise_random(out[i,:,:])
        if self.ma:
            out = np.moveaxis(ma_batch(out, self.ma_len), -1, 0)
        return out,  self.y[indexes,:]
