import numpy as np


def dm_to_numpy(loader, n=None):
    print(len(loader))
    data = [xy for xy in loader]
    x = np.vstack([xy[0] for xy in data])
    y = np.concatenate([xy[1] for xy in data])
    x = x[:n]
    y = y[:n]
    n = x.shape[0]
    x = x.reshape(n, -1)
    y = y.reshape(n, -1)  # TODO double-check inverse operation for pred

    print(x.shape, y.shape)
    return x, y
