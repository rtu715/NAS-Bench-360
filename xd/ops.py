import math
import pdb
from functools import partial
from itertools import product
import torch
import torch.fft
from torch import nn
from torch.nn import functional as F
from .utils import complex_convert, complex_matmul, int2tuple


def Conv(dims):
    '''returns PyTorch convolution module of specified dimension'''

    return getattr(nn, 'Conv'+str(dims)+'d')


def AvgPool(dims):
    '''returns PyTorch average pooling module of specified dimension'''

    return getattr(nn, 'AvgPool'+str(dims)+'d')


class Diagonal(nn.Module):

    def __init__(self, diag):

        super().__init__()
        diag = torch.Tensor(diag)
        if all(diag == torch.ones(len(diag))):
            diag = None
        self.register_buffer('diag', diag)

    def forward(self, input):

        if self.diag is None:
            return input
        return input * self.diag


class Fourier(nn.Module):

    def __init__(self, inv=False, normalized=False):

        super().__init__()
        self.inv = inv
        self.norm = 'ortho' if normalized else 'backward'

    def forward(self, input):

        func = torch.fft.ifft if self.inv else torch.fft.fft
        return func(input, norm=self.norm)


class TensorProduct(nn.Module):
    '''generalizes torch_butterfly.combine.TensorProduct to handle products of any length'''

    def getmap(self, dim):
        '''return map associated with given dimension'''

        return getattr(self, 'map'+str(dim))

    def setmap(self, dim, attr):
        '''set map associated with given dimension to the given attribute'''

        setattr(self, 'map'+str(dim), attr)
        self.maps[dim-1] = attr

    def __init__(self, *maps):
        '''
        Args:
            maps: any number of torch_butterfly.butterfly.Butterfly objects; also handles torch_butterfly.combine.TensorProduct objects
        '''

        super().__init__()
        self.register_buffer('dummy', torch.Tensor([]))
        try:
            maps = [maps[0].map1, maps[0].map2]
        except (AttributeError, IndexError):
            pass

        self.maps = [None] * len(maps)
        for i, m in enumerate(maps):
            self.setmap(i+1, m)

    def forward(self, input):

        for i, m in enumerate(self.maps):
            input = m(input.transpose(-1, -i-1)).transpose(-1, -i-1)
        return input

    def device(self):
        '''returns device associated with the first element in the product'''

        for p in self.parameters():
            return p.device
        for b in self.buffers():
            return b.device
        return self.dummy.device


class FNO(nn.Module):

    @staticmethod
    def ifftn(*args, **kwargs):
        return torch.fft.ifftn(*args, **kwargs).real

    def __init__(self, in_channels, out_channels, modes, hermitian=True, pad=False):

        super(FNO, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dims = 1 if type(modes) == int else len(modes)
        self.modes = int2tuple(modes, length=self.dims)
        self.scale = 1. / (in_channels * out_channels)
        self.hermitian = hermitian
        self.pad = pad
        self.fft = torch.fft.rfftn if self.hermitian else torch.fft.fftn
        self.ifft = torch.fft.irfftn if self.hermitian else self.ifftn
        kernel_size = [2*m for m in self.modes[:-1]] + [(2-hermitian)*self.modes[-1]]
        self.weight = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, *kernel_size, dtype=torch.complex64))

    def get_slices(self, size=None):

        if self.hermitian:
            size = [2*m for m in self.modes[:-1]] if size is None else size[:-1]
            modes = self.modes[:-1]
            end = [slice(self.modes[-1])]
        else:
            size = [2*m for m in self.modes] if size is None else size
            modes = self.modes
            end = []
        for bits in product(*[range(2)] * (self.dims-self.hermitian)):
            yield [slice(None), slice(None)] + [slice(n-m, n) if b else slice(m) for b, n, m in zip(reversed(bits), size, modes)] + end

    def forward(self, x):

        if self.pad:
            unpad = [slice(None)] * 2 + [slice(s) for s in x.shape[2:]]
            s = [2 ** math.ceil(math.log2(s)) for s in x.shape[2:]]
            size = s[:-1] + [s[-1]//2+1 if self.hermitian else s[-1]]
        else:
            unpad, s = [slice(None)], None
            size = list(x.shape[2:-1]) + [x.shape[-1]//2+1 if self.hermitian else x.shape[-1]]
        x_ft = self.fft(x, s=s, dim=tuple(range(-self.dims, 0)), norm='ortho')
        out_ft = torch.zeros(len(x), self.out_channels, *size, dtype=torch.complex64, device=x.device)
        for xslices, wslices in zip(self.get_slices(size), self.get_slices()):
            out_ft[xslices] = complex_matmul(x_ft[xslices].permute(*range(2, 2+self.dims), 0, 1), 
                                             self.weight[wslices].permute(*range(2, 2+self.dims), 1, 0)).permute(-2, -1, *range(self.dims))
        return self.ifft(out_ft, dim=tuple(range(-self.dims, 0)), norm='ortho')[unpad]


def get_fno(sconv, **kwargs):

    modes = []
    i = 0
    while True:
        i += 1
        try:
            modes.append(getattr(sconv, 'modes'+str(i)))
        except AttributeError:
            break

    fno = FNO(sconv.in_channels, sconv.out_channels, modes, **kwargs)
    for i, slices in enumerate(fno.get_slices()):
        weight = complex_convert(getattr(sconv, 'weights'+str(i+1))).transpose(0, 1)
        fno.weight.data[slices] = weight
    if not fno.hermitian:
        fno.weight.data = fno.hermitian_pad(xd.weight[[slice(None)]*(fno.dims+1)+[slice(modes[-1])]])
    return fno


if __name__ == '__main__':

    import sys; sys.path.insert(0, './fourier_neural_operator')
    from fourier_1d import SpectralConv1d
    from fourier_2d import SpectralConv2d
    from fourier_3d import SpectralConv3d_fast

    chan, modes = 8, 12
    for dims in range(1, 4):

        x = torch.normal(0., 1., (2, chan, *[48]*dims))
        sconv = {1: SpectralConv1d, 
                 2: SpectralConv2d,
                 3: SpectralConv3d_fast}[dims](chan, chan, *[modes]*dims)
        true = sconv(x)
        fno = get_fno(sconv)
        test = fno(x)
        print('dim:', dims, 
              '\terr:', (torch.norm(test-true) / torch.norm(true)).item())
