import torch
if int(torch.__version__.split('.')[1]) > 7:
    from torch import matmul as complex_matmul
else:
    from torch_butterfly.complex_utils import complex_matmul


def int2tuple(int_or_tuple, length=2):
    '''converts bools, ints, or slices to tuples of the specified length via repetition'''

    if type(int_or_tuple) in {bool, int, slice}:
        return tuple([int_or_tuple] * length)
    assert len(int_or_tuple) == length, "tuple must have length " + str(length)
    return int_or_tuple


def complex_convert(x):

    slices = [slice(None) for _ in range(len(x.shape)-1)]
    return torch.complex(x[slices+[0]], x[slices+[1]])
