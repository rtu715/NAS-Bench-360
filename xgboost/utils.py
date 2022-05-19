import numpy as np


def dm_to_numpy(loader, n=None):
    x = np.vstack([xy[0] for xy in loader])
    y = np.concatenate([xy[1] for xy in loader])
    x = x[:n]
    y = y[:n]
    n = x.shape[0]
    x = x.reshape(n, -1)
    return x, y
