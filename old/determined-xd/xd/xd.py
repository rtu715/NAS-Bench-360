import math
import pdb
from copy import deepcopy
from functools import partial
from itertools import product
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint as torch_checkpoint
from torch_butterfly import Butterfly
from torch_butterfly.combine import diagonal_butterfly
from torch_butterfly.complex_utils import Real2Complex, Complex2Real
from torch_butterfly.permutation import FixedPermutation, bitreversal_permutation, perm2butterfly
from torch_butterfly.special import fft, ifft, fft2d, ifft2d
from .ops import AvgPool, Conv, Diagonal, FNO, Fourier, TensorProduct
from .utils import complex_convert, complex_matmul, int2tuple


def complex_norm(X):
    '''returns norm of the concatenated real and imaginary components of given Tensor'''

    return torch.sqrt(X.real.square().sum() + X.imag.square().sum())

def special_dist(X, Y, relative=False, scaled=False):
    '''returns distance between two Tensors
    Args:
        X: first tensor
        Y: second tensor
        relative: compute relative distance (normalizes by Y)
        scaled: returns minimum distance of arbitrary scaling of X
    Returns:
        singleton Tensor
    '''
    
    if scaled:
        denom = X.square().sum()
        if denom:
            X *= (X * Y).sum() / denom
    return (X-Y).norm() / Y.norm() if relative else (X-Y).norm()

def complex_dist(X, Y, **kwargs):
    '''complex version of 'special_dist' method'''

    return special_dist(torch.cat((X.real, X.imag)), torch.cat((Y.real, Y.imag)), **kwargs)


def fourier_diag(*nd, diags=None, inv=False, diag_first=True, with_br_perm=False, fixed=False, **kwargs):
    '''returns n-dimensional FFT Butterfly matrix multiplied by a diagonal matrix
    Args:
        nd: input sizes of each dimension
        diags: torch.Tensor vectors specifying diagonals; if None uses the identity
        inv: return inverse FFT
        diag_first: returns FFT * diagonal; if False returns diagonal * FFT
        with_br_perm: uses bit-reversal permutation
        fixed: use fixed modules instead of k-matrices
        kwargs: passed to torch_butterfly.special.fft
    Returns:
        TensorProduct object
    '''

    if fixed:
        br_first = kwargs.get('br_first')
        if br_first is None:
            br_first = True
        else:
            del kwargs['br_first']
        maps = []
        for n, diag in zip(nd, diags):
            mods = [Fourier(inv=inv, **kwargs)]
            if not with_br_perm:
                perm = FixedPermutation(torch.LongTensor(bitreversal_permutation(n)))
                mods = [perm]+mods if br_first else mods+[perm]
            if not diag is None:
                mod = Diagonal(diag)
                mods = [mod]+mods if diag_first else mods+[mod]
            maps.append(nn.Sequential(*mods))
        return TensorProduct(*maps)

    kwargs['with_br_perm'] = with_br_perm

    if diags is None:
        if len(nd) == 1:
            return TensorProduct(ifft(*nd, **kwargs) if inv else fft(*nd, **kwargs))
        if len(nd) == 2:
            return TensorProduct(ifft2d(*nd, **kwargs) if inv else fft2d(*nd, **kwargs))
        diags = [torch.ones(n) for n in nd]

    assert set(kwargs.keys()).issubset({'normalized', 'br_first', 'with_br_perm'}) and not with_br_perm, \
            "invalid kwargs when using diags or >2 dims"
    func = ifft if inv else fft
    return TensorProduct(*(diagonal_butterfly(func(n, **kwargs), diag, diag_first=diag_first) for n, diag in zip(nd, diags)))

fftnd_diag = partial(fourier_diag, inv=False)
ifftnd_diag = partial(fourier_diag, inv=True)


class XD(nn.Module):
    '''XD-Operation module for all dimensions'''

    r2c = Real2Complex()
    c2r = Complex2Real()
    fourier_K = partial(ifftnd_diag, normalized=True, br_first=True, diag_first=False)
    fourier_L = partial(fftnd_diag, normalized=False, br_first=False, diag_first=True)
    fourier_M = partial(fftnd_diag, normalized=True, br_first=False, diag_first=True)
    
    def get_fourier(self, kmatrix_name, *nd, **kwargs):
        return getattr(self, 'fourier_'+kmatrix_name)(*nd, **kwargs)

    @staticmethod
    def diag_K(in_size, skip_interval):

        diag = torch.zeros(in_size)
        diag[::skip_interval] = 1.0
        return diag

    @staticmethod
    def diag_L(in_size, kernel_size):

        diag = torch.zeros(in_size)
        half = kernel_size // 2
        diag[:half] = 1.0
        diag[in_size-half:] = 1.0
        if 2 * half < kernel_size:
            diag[half] = 1.0
        return diag

    @staticmethod
    def diag_M(in_size, crop_init):

        diag = torch.ones(in_size)
        diag[crop_init] = 0.0
        return diag

    def _perturb(self, tensor, perturb):
        '''perturbs tensor with random noise of the specified deviation'''

        return tensor + torch.normal(0., perturb, tensor.shape) if perturb else tensor

    def _circular_pad(self, weight):
        '''circularly pads filter weights'''

        for dim, n, k, p in zip(range(1, self.dims+1), self.nd, self.kd, self.pd):
            if dim >= 2 and self.weight.dtype == torch.complex64:
                weight = torch.cat([F.pad(weight[...,:n//2].flip([-dim]), (*[0]*(2*dim-1), n-k)).roll(-p, dims=-dim),
                                    F.pad(weight[...,-n//2:].flip([-dim]), (*[0]*(2*dim-1), n-k)).roll(-p+1, dims=-dim)],
                                    dim=self.dims+1)
            else:
                weight = F.pad(weight.flip([-dim]), (*[0]*(2*dim-1), n-k)).roll(-p, dims=-dim)
        return self.perm(weight)

    @staticmethod
    def _atrous_permutation(n, k, d):
        '''computes permutation of circularly padded filter weights to match circularly padded dilated filter weights'''

        perm = torch.arange(n).roll(k//2-k).flip(0)
        for i in range(k-1, 0, -1):
            perm[i], perm[d*i] = perm[d*i].item(), perm[i].item()
        perm[:d*(k-1)+1] = perm[:d*(k-1)+1].flip(0)
        return perm.roll(-((d*(k-1)+1)//2))

    @staticmethod
    def _offset_insert(output, kernel):
        '''inserts filter into the middle of a larger filter'''

        sizes = kernel.shape[2:]
        offsets = [(o-k+int(o % 2 == 1)) // 2 for o, k in zip(output.shape[2:], sizes)]
        output[[slice(None), slice(None)] + [slice(offset, offset+size) for offset, size in zip(offsets, sizes)]] = kernel
        return output

    def _parse_size(self, arch_init):

        if '_' in arch_init:
            xsplit = arch_init.split('_')[-1].split('x')
            if len(xsplit) != self.dims:
                assert len(xsplit) == 1, "cannot parse architecture initialization"
                xsplit = xsplit[:1] * self.dims
            return tuple(int(k) for k in xsplit)

    def _parse_init(self, arch_init, max_kernel_size, padding, arch_shape, dilation_init, _swap):

        max_kernel_size = int2tuple(max_kernel_size, length=self.dims)
        kd_init = list(reversed(max_kernel_size))
        skips = [1] * self.dims

        size = self._parse_size(arch_init)
        if not size is None:
            assert all(k == s for k, s in zip(max_kernel_size, size)) or not 'fno' in arch_init, "max kernel size must match arch_init if using fno"
            if 'conv' in arch_init or 'pool' in arch_init or 'fno' in arch_init:
                kd_init = tuple(reversed(size))
                max_kernel_size = tuple(max(k, s) for k, s in zip(max_kernel_size, size))
            else:
                skips = tuple(reversed(size))

        if _swap:
            unpadding = self.unpadding
        elif arch_init == arch_shape or arch_shape is None:
            if padding is None and (arch_init == 'ortho' or 'skip' in arch_init or 'fno' in arch_init):
                unpadding = [(0, n) for n in self.in_size]
            else:
                unpadding = []
                for d, k, m, n, p in zip(reversed(dilation_init),
                                         reversed(kd_init), 
                                         max_kernel_size,
                                         self.in_size, 
                                         ((d*(s-1)+1) // 2 for s, d in zip(reversed(kd_init), reversed(dilation_init))) if padding is None else int2tuple(padding, length=self.dims)):
                    # handles padding to match corresponding PyTorch modules
                    a, b = (d*(k-1)) // 2 - p, n - (d*(k-1)+1) // 2 + p
                    if d > 1 and (m-k) % 2:
                        a, b = a+1 - 2*int(not k % 2), b+1 - 2*int(not k % 2)
                    assert 0 <= a <= b <= n, "invalid padding"
                    unpadding.append((a, b))
        else:
            _, _, _, _, _, unpadding = self._parse_init(arch_shape, max_kernel_size, padding, arch_shape, dilation_init, _swap)

        return max_kernel_size, \
                kd_init, \
                skips, \
                any(name in arch_init for name in ['conv', 'pool', 'skip', 'fno']), \
                any(name in arch_init for name in ['pool', 'skip']), \
                unpadding

    def __init__(self, in_size, in_channels, out_channels, arch_init='ortho', weight_init=nn.init.kaiming_normal_, kmatrix_depth=1, base=2, max_kernel_size=1, padding=None, stride=1, arch_shape=None, weight=None, global_biasing='additive', channel_gating='complete', perturb=0.0, crop_init=slice(0), dilation_init=1, padding_mode='circular', bias=None, checkpoint=False, fourier_position=-1, fixed=False, complex_weight=False, herm_pad=False, perm_fix=False, low_freq=False, _swap=False):
        '''
        Args:
            in_size: input size
            in_channels: number of input channels
            out_channels: number of output_channels
            arch_init: 'ortho' or $OPTYPE (e.g. 'skip') or $OPTYPE'_'$KERNELSIZE (e.g. 'conv_3x3')
            weight_init: function that initializes weight tensor
            kmatrix_depth: depth of each kmatrix
            base: base to use for kmatrix (must be 2 or 4)
            max_kernel_size: maximum kernel size
            padding: determines padding; by default sets padding according to arch_init 
            stride: governs subsampling
            arch_shape: architecture that determines the output shape; uses arch_init by default
            weight: model weights
            global_biasing: 'additive' or 'interp' or False
            channel_gating: 'complete' or 'interp' or False
            perturb: scale of perturbation to arch params
            crop_init: input slice(s) to crop
            dilation_init: kernel dilation at initialization
            padding_mode: 'circular' or 'zeros'; for 'zeros' will adjust in_size as needed
            bias: optional bias parameter
            checkpoint: apply checkpointing to kmatrix operations
            fourier_position: where to put each Fourier matrix when warm starting; -1 applies it last
            fixed: use fixed operation instead of kmatrix 
            complex_weight: use complex model weights
            herm_pad: constrain weights of last dimension to be Hermitian
            perm_fix: fix dilation/bitreversal permutation for middle kmatrix
            low_freq: zero out high frequency components
        '''

        if not _swap:
            # '_swap' variable allows for fast re-initialization of a module; useful for computing metrics
            super(XD, self).__init__()
            self._init_args = (in_size, in_channels, out_channels)
            self._init_kwargs = {
                                 'arch_shape': arch_init, 
                                 'padding': padding, 
                                 'crop_init': crop_init, 
                                 'dilation_init': dilation_init, 
                                 'padding_mode': padding_mode, 
                                 'checkpoint': checkpoint,
                                 'fourier_position': fourier_position,
                                 'fixed': fixed,
                                 'complex_weight': complex_weight,
                                 'herm_pad': herm_pad,
                                 'perm_fix': perm_fix,
                                 }
        assert base in {2, 4}, "'base' must be 2 or 4"
        assert global_biasing in {'additive', 'interp', False}, "invalid value for 'global_biasing'"
        assert channel_gating in {'complete', 'interp', False}, "invalid value for 'channel_gating'"

        self.checkpoint = checkpoint
        self.base = base
        self.chan = (out_channels, in_channels)
        self.depth = int2tuple(kmatrix_depth, length=3)
        self.dims = 2 if type(in_size) == int else len(in_size)
        in_size = int2tuple(in_size, length=self.dims)
        if padding_mode == 'zeros':
            # increases effective input size if required due to zero-padding
            padding = int2tuple(0 if padding is None else padding, length=self.dims)
            in_size = tuple(n+2*p for n, p in zip(in_size, padding))
            self.zero_pad = tuple(sum(([p, p] for p in padding), []))
            padding = [0] * self.dims
        else:
            self.zero_pad = ()
        self.in_size = tuple(2 ** math.ceil(math.log2(n)) for n in in_size)
        crop_init = int2tuple(crop_init, length=self.dims)
        dilation_init = tuple(reversed(int2tuple(dilation_init, length=self.dims)))
        self.max_kernel_size, kd_init, skips, fourier_init, diagonal_init, self.unpadding = self._parse_init(arch_init, 
                                                                                                             max_kernel_size, 
                                                                                                             padding, 
                                                                                                             arch_shape, 
                                                                                                             dilation_init, 
                                                                                                             _swap)
        zeroL = diagonal_init and global_biasing == 'additive'
        self.nd = tuple(reversed(self.in_size))
        self.kd = tuple(reversed(self.max_kernel_size))
        self.pd = tuple(k // 2 for k in self.kd)
        self.stride = int2tuple(stride, length=self.dims)
        if all(s == 1 for s in self.stride):
            self.subsample = nn.Sequential()
        else:
            assert self.dims <= 3, "must have stride 1 if using >3 dims" # TODO: handle stride>1 for >3 dimensional XD-op
            self.subsample = AvgPool(self.dims)(kernel_size=[1]*self.dims, stride=self.stride)

        self.herm_pad = herm_pad
        if herm_pad:
            assert self.max_kernel_size[-1] % 2 == 0, "last kernel size must be even if using Hermitian padding"
            self.max_kernel_size = tuple(list(self.max_kernel_size[:-1]) + [self.max_kernel_size[-1] // 2])
        if not _swap:
            tensor = torch.Tensor(out_channels, in_channels, *self.max_kernel_size)
            if complex_weight:
                tensor = torch.complex(tensor, tensor)
            self.weight = nn.Parameter(tensor)
            weight_init(self.weight)
        if not weight is None:
            if type(weight) == nn.Parameter and self.weight.shape == weight.shape:
                self.weight = weight
            else:
                self._offset_insert(self.weight.data, weight.data.to(self.weight.device))
        self.bias = nn.Parameter(bias) if type(bias) == torch.Tensor else bias

        channels = min(self.chan)
        inoff, outoff = int(0.5*(in_channels-channels)), int(0.5*(out_channels-channels))
        if not _swap:
            self.register_buffer('diag', None, persistent=False)
            self.register_buffer('kron', None, persistent=False)
            self.register_buffer('_one', self.r2c(torch.ones(1)))
            self.register_buffer('_1', self.r2c(torch.ones(self.chan)))
            self.register_buffer('_I', self.r2c(torch.zeros(self.chan)))
            self._I[outoff:outoff+channels,inoff:inoff+channels] = torch.eye(channels)

        fixed = int2tuple(fixed, length=3)
        self.low_freq = low_freq
        if low_freq:
            assert fixed[0] == fixed[2], "K and M must be either both fixed or not fixed when using low_freq"
        brs, perms = [], []
        for (kmatrix_name, diags), depth, fpos, fix in zip([('K', [self.diag_K(n, s) for n, s in zip(self.nd, skips)]), # handles strides
                                                            ('L', [torch.zeros(n) if zeroL else torch.ones(n) if complex_weight else self.diag_L(n, k) for n, k in zip(self.nd, kd_init)]), # handles kernel size limits
                                                            ('M', [self.diag_M(n, c) for n, c in zip(self.nd, crop_init)])], # handles input cropping
                                                           self.depth, 
                                                           int2tuple(fourier_position, length=3),
                                                           fixed):

            if kmatrix_name != 'L' and low_freq and not (fix or perm_fix):
                assert not depth % 2, "kmatrix "+kmatrix_name+" must have even depth"
            else:
                assert depth % 2 or (kmatrix_name == 'L' and 'fno' in arch_init) or fix or not fourier_init, "kmatrix "+kmatrix_name+" must have odd depth"

            if _swap:
                kmatrix = getattr(self, kmatrix_name)
            else:
                kmatrix_kwargs = {
                                  'bias': False, 
                                  'increasing_stride': kmatrix_name == 'K' and (fix or perm_fix or not low_freq), 
                                  'complex': True, 
                                  'init': 'identity' if fourier_init else arch_init,
                                  'nblocks': depth,
                                  }
                kmatrix = TensorProduct(*(Butterfly(n, n, **kmatrix_kwargs) for n in self.nd))
            if fourier_init:
                fourier_kmatrix = self.get_fourier(kmatrix_name, 
                                                   *self.nd, 
                                                   diags=[self._perturb(diag if d == 1 else torch.ones(diag.shape), perturb) for d, diag in zip(dilation_init, diags)],
                                                   fixed=fix)
                if (kmatrix_name == 'L' and any(d > 1 for d in dilation_init)) or (kmatrix_name == 'K' and low_freq and not perm_fix):
                    fpos = max(2, depth+fpos if fpos < 0 else fpos)
                elif (kmatrix_name == 'M' and low_freq and not perm_fix):
                    fpos = min(depth-4, depth+fpos if fpos < 0 else fpos)
                for dim, d, k, n in zip(range(1, self.dims+1), dilation_init, self.kd, self.nd):
                    perm = None
                    if kmatrix_name == 'L':
                        if 'fno' in arch_init and not low_freq:
                            assert fix or perm_fix or depth >= 2, "using fno requires depth at least (1, 2, 1)"
                            perm = torch.LongTensor(bitreversal_permutation(n))
                        elif d > 1:
                            assert fix or perm_fix or depth >= 3, "using dilation > 1 requires depth at least (1, 3, 1)"
                            perm = self._atrous_permutation(n, k, d)
                        if perm_fix and (not fix) and (not perm is None):
                            perms.append(FixedPermutation(perm))
                            perm = None
                    elif low_freq:
                        assert fix or perm_fix or depth >= 4, "using low_freq requires depth at least (4, 1, 4)"
                        perm = torch.LongTensor(bitreversal_permutation(n))
                    if perm_fix and (not fix) and (not perm is None):
                        if len(brs) < self.dims:
                            brs.append(FixedPermutation(perm))
                        perm = None
                    if fix:
                        if perm is None and not (kmatrix_name == 'L' and 'fno' in arch_init):
                            kmatrix.setmap(dim, fourier_kmatrix.getmap(dim))
                        else:
                            mods = [] if perm is None else [FixedPermutation(perm)]
                            if kmatrix_name == 'K' or not 'fno' in arch_init:
                                mods.append(fourier_kmatrix.getmap(dim))
                            elif kmatrix_name == 'M':
                                mods = [fourier_kmatrix.getmap(dim)] + mods
                            if not diags[dim-1] is None:
                                mods = [Diagonal(diags[dim-1])] + mods
                            kmatrix.setmap(dim, nn.Sequential(*mods))
                    else:
                        if not perm is None:
                            twiddle = diagonal_butterfly(perm2butterfly(perm, complex=True),
                                                         diags[dim-1], diag_first=True).twiddle.data.to(kmatrix.device())
                            if kmatrix_name == 'M':
                                kmatrix.getmap(dim).twiddle.data[:,-2:] = twiddle
                            else:
                                kmatrix.getmap(dim).twiddle.data[:,:2] = twiddle
                        if kmatrix_name != 'L' or not 'fno' in arch_init:
                            kmatrix.getmap(dim).twiddle.data[0,fpos] = fourier_kmatrix.getmap(dim).twiddle.data[0,0].to(kmatrix.device())
            if base == 4 and not fix:
                for dim in range(1, self.dims+1):
                    kmatrix.setmap(dim, kmatrix.getmap(dim).to_base4())
            setattr(self, kmatrix_name, kmatrix)

        self.br = TensorProduct(*brs)
        self.perm = TensorProduct(*perms)

        self.global_biasing = global_biasing
        filt = self._offset_insert(torch.zeros(1, 1, *self.max_kernel_size),
                                   torch.ones(1, 1, *kd_init) / np.prod(kd_init) if 'pool' in arch_init else torch.ones(1, 1, *[1]*self.dims))
        if self.global_biasing == 'additive':
            if diagonal_init:
                L = self.get_fourier('L', *self.nd, diags=[self.diag_L(n, k) for n, k in zip(self.nd, kd_init)])
                b = L(self.r2c(self._circular_pad(filt)))
            else:
                b = self.r2c(torch.zeros(1, 1, *self.in_size))
        elif self.global_biasing == 'interp':
            if diagonal_init:
                b = self.r2c(torch.cat((torch.ones(1), filt.flatten())))
            else:
                b = self.r2c(torch.zeros(1 + np.prod(self.max_kernel_size)))
        else:
            b = self.r2c(torch.Tensor(0))
        if _swap:
            self.b.data = b.to(self.b.device)
        else:
            self.register_parameter('b', nn.Parameter(b))

        self.channel_gating = channel_gating
        if self.channel_gating == 'complete':
            if diagonal_init:
                C = self.r2c(torch.zeros(self.chan))
                C[outoff:outoff+channels,inoff:inoff+channels] = torch.eye(channels)
            else:
                C = self.r2c(torch.ones(self.chan))
        elif self.channel_gating == 'interp':
            C = self.r2c(torch.Tensor([float(diagonal_init)]))
        else:
            C = self.r2c(torch.Tensor(0))
        if _swap:
            self.C.data = C.to(self.C.device)
        else:
            self.register_parameter('C', nn.Parameter(C))

        self.to(self.device())

    def _checkpoint(self, func, *args):

        if self.checkpoint:
            return torch_checkpoint.checkpoint(func, *args)
        return func(*args)

    @staticmethod
    def hermitian_pad(weight):

        dims = len(weight.shape)-2
        modes = weight.shape[-1]
        return torch.cat([weight, F.pad(torch.flip(torch.conj(weight[...,:modes]), dims=list(range(-dims, 0)))[...,1:modes], (0, 1))], dim=dims+1)

    def _diag(self, weight=None, _batch=False):

        if weight is None and not self.training:
            # uses cached diagonal matrix in evaluation mode
            return self.diag
        weight = self.weight if weight is None else weight
        if weight.dtype == torch.float32:
            weight = self.r2c(weight)
        if self.herm_pad:
            weight = self.hermitian_pad(weight)

        if self.global_biasing == 'interp':
            diag = self._checkpoint(self.L, self._circular_pad((self._one-self.b[0]) * weight + self.b[0] * self.b[1:].reshape(1, 1, *self.max_kernel_size)))
        else:
            diag = self._checkpoint(self.L, self._circular_pad(weight))
            if self.global_biasing == 'additive':
                diag = diag + self.b

        if _batch:
            # handles the case where 'weight' is a batch of different weight filter Tensors; useful for 'averaged' metric
            diag = diag.permute(*range(1, 3+self.dims), 0)
            dims = self.dims + 1
        else:
            dims = self.dims

        if self.channel_gating == 'complete':
            diag = (self.C.flatten().reshape(-1, *[1]*dims) * diag.flatten(0, 1)).reshape(diag.shape)
        elif self.channel_gating == 'interp':
            diag = (((self._one - self.C) * self._1 + self.C * self._I).flatten().reshape(-1, *[1]*dims) * diag.flatten(0, 1)).reshape(diag.shape)

        if _batch:
            return diag.permute(-1, *range(2+self.dims))
        return diag

    def train(self, mode=True):
        '''computes cached diagonal matrix before entering evaluation mode'''

        self.diag = None if mode else self._diag()
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def _low_freq(self):

        start = [slice(None), slice(None)]
        freqs = [k//2 for k in self.max_kernel_size[:-1]] + [self.max_kernel_size[-1] // (2-self.herm_pad)]
        for bits in product(*[range(2)] * self.dims):
            if self.dims == 1:
                n = self.in_size[0]
                yield start + [slice(n-freqs[0], n) if bits[0] else slice(freqs[0])]
            else:
                yield start + [slice(n-f+bits[0], n) if b else slice(f+bits[0]) 
                               for b, n, f in zip(reversed(bits), self.in_size, freqs)]

    def forward(self, x, weight=None):

        x = F.pad(x, self.zero_pad)

        pad, unpad = [], [slice(None), slice(None)]
        center = not self.weight.dtype == torch.complex64
        for xn, n, (a, b) in zip(x.shape[2:], self.in_size, self.unpadding):
            if center:
                p1 = (n-xn) // 2
                p2 = (n-xn) // 2
                p1 += p1+p2 < n-xn
                pad = [p1, p2] + pad
                unpad.append(slice(a+p1, b-p2))
            else:
                pad = [0, n-xn] + pad
                unpad.append(slice(a, b-n+xn))
        x = F.pad(x, pad)

        x = self._checkpoint(self.M, self.r2c(x))
        diag = self._diag(weight=weight)
        if self.low_freq:
            x = self.br(x)
            out = torch.zeros(*x.shape, dtype=torch.complex64, device=x.device)
            for slices in self._low_freq():
                out[slices] = complex_matmul(x[slices].permute(*range(2, 2+self.dims), 0, 1), diag[slices].permute(*range(2, 2+self.dims), 1, 0)).permute(-2, -1, *range(self.dims))
            x = self.br(out)
        else:
            x = complex_matmul(x.permute(*range(2, 2+self.dims), 0, 1), diag.permute(*range(2, 2+self.dims), 1, 0)).permute(-2, -1, *range(self.dims))
        x = self.c2r(self._checkpoint(self.K, x))
        x = self.subsample(x[unpad])
        if self.bias is None:
            return x
        return x + self.bias.reshape(1, *self.bias.shape, *[1]*self.dims)

    def _kron(self):

        if self.kron is None:
            I = torch.eye(self.in_size[0]).to(self.device())
            for size in self.in_size[1:]:
                I = torch.kron(I, torch.eye(size).to(self.device()))
            self.kron = self.r2c(I.reshape(np.prod(self.in_size), *self.in_size))
        return self.kron

    def _kmatrix_expansion(self, *args, approx=None, perm=None):
        '''computes (transpose of) expanded dense K-matrix, or a projection onto randomly sampled standard basis vectors'''
        
        prod = np.prod(self.in_size)
        approx = prod if approx is None else min(approx, prod)
        if approx == prod and perm is None:
            I = self._kron()
        else:
            perm = torch.randperm(prod) if perm is None else perm
            I = self._kron()[perm[:approx]] / np.sqrt(approx)
        return [kmatrix(I) for kmatrix in args]

    def _op_expansion(self, *args, weight=None, _batch=False, **kwargs):
        '''computes expanded dense XD-op, or a projection onto randomly sampled standard basis vectors'''

        prod = np.prod(self.in_size)
        output = []
        for xd, M in zip(args, self._kmatrix_expansion(*(xd.M for xd in args), **kwargs)):
            chan = ([weight.shape[0]] if _batch else []) + list(xd.chan)
            Lw = xd._diag(weight=weight, _batch=_batch).reshape(*chan, 1, prod)
            LwM = (Lw * M.reshape(len(M), prod)).reshape(*chan, len(M), *xd.in_size)
            output.append(self.c2r(xd.K(LwM)))
        return output

    def triu(self, offset=0, approx=None):

        assert self.dims == 1, "upper triangular penalty only implemented for 1d XD-ops"
        size = self.in_size[0]
        perm = torch.arange(offset, size)
        if approx is None or approx >= size-offset:
            approx = size-offset
        else:
            perm = perm[torch.randperm(size-offset)]
        perm = torch.cat([perm, torch.LongTensor([0])])
        KLwM = self._op_expansion(self, approx=approx, perm=perm)[0]
        perm = perm[:approx]
        return 0.5 * KLwM[:,:,(torch.arange(size*size) % size).reshape(size, size)[perm] <= perm[:,None]-offset].square().sum()

    def penalty(self, other=None, **kwargs):
        '''returns half the squared Frobenius norm of the expanded XD-op'''

        if other is None:
            return 0.5 * self._op_expansion(self, **kwargs)[0].square().sum()
        return 0.5 * torch.sub(*self._op_expansion(self, other, **kwargs)).square().sum()

    def frobenius(self, other=None, relative=False, scaled=False, **kwargs):
        '''returns Frobenius norm of the expanded XD-op'''

        if other is None:
            return self._op_expansion(self, **kwargs)[0].norm()
        return special_dist(*self._op_expansion(self, other, **kwargs), relative=relative, scaled=scaled)

    def averaged(self, batched_weights, other, relative=False, scaled=False, _cache=[], **kwargs):
        '''returns Frobenius distance between XD-op and another compatible XD-op, averaged over different weights'''

        if _cache:
            expansion = [_cache[0], self._op_expansion(other, weight=batched_weights, perm=_cache[1], **kwargs)[0]]
        else:
            perm = torch.randperm(np.prod(self.in_size))
            expansion = self._op_expansion(self, other, weight=batched_weights, perm=perm, **kwargs)
            _cache.extend([expansion[0], perm])
        return sum(special_dist(X, Y, relative=relative, scaled=scaled) for X, Y in zip(*expansion)) / len(batched_weights)

    def euclidean(self, other=None, **kwargs):
        '''returns Euclidean norm of architecture parameters'''

        if other is None:
            return complex_norm(torch.cat([p.data.flatten() for p in self.arch_params()]))
        return complex_dist(torch.cat([p.data.flatten() for p in self.arch_params()]), 
                            torch.cat([q.data.flatten() for q in other.arch_params()]), 
                            **kwargs)

    def device(self):
        '''returns device associated with the 'K' architecture parameter'''

        return self.K.device()

    def distance_from(self, src='conv', metric='frobenius', samples=40, weight_init=nn.init.kaiming_normal_, relative=False, scaled=False, approx=None, **kwargs):
        '''
        Args:
            src: $OPTYPE (e.g. 'skip') or $OPTYPE'_'$KERNELSIZE (e.g. 'conv_3x3')
            metric: 'frobenius' or 'euclidean' or 'averaged'
            samples: number of samples to use for 'averaged'
            weight_init: weight initialization method to use for 'averaged'
            crop_init: passed to self.__init__
            relative: computes relative error
            scaled: computes error after allowing arbitrary scaling
            approx: number of basis vectors for 'frobenius' and 'averaged' metrics
            kwargs: passed to XD.__init__
        Returns:
            distance to closest named operation according to given metric
        '''

        if metric == 'averaged':
            batched_weights = torch.stack([weight_init(self.weight.clone()).to(self.device()) for _ in range(samples)])
            cache = []
            func = partial(self.averaged, batched_weights, relative=relative, scaled=scaled, approx=approx, _cache=cache, _batch=True)
        elif metric == 'euclidean':
            func = partial(self.euclidean, relative=relative, scaled=scaled)
        elif metric == 'frobenius':
            func = partial(self.frobenius, relative=relative, scaled=scaled, approx=approx)
        else:
            raise NotImplementedError
        size = self._parse_size(src)
        sizes = list(product(*(range(1, k+1) for k in reversed(self.kd)))) if size is None else [size]

        xd_kwargs = {
                      'kmatrix_depth': self.depth if metric == 'euclidean' else (1, 3, 1) if sum(int2tuple(self._init_kwargs['dilation_init'], length=self.dims)) > self.dims else 1,
                      'base': self.base if metric == 'euclidean' else 2,
                      'max_kernel_size': self.max_kernel_size,
                      'stride': self.stride,
                      'weight': self.weight.to(self.device()) if metric == 'frobenius' and 'conv' in src else None,
                      'global_biasing': self.global_biasing,
                      'channel_gating': self.channel_gating,
                      }
        xd_kwargs.update(self._init_kwargs)
        xd_kwargs.update(kwargs)
        xd = XD(*self._init_args,
                  arch_init='_'.join([src, 'x'.join(str(s) for s in sizes[0])]),
                  **xd_kwargs).to(self.device())

        with torch.no_grad():
            dist = func(xd, **kwargs)
            for size in sizes:
                xd.__init__(*self._init_args,
                             arch_init='_'.join([src, 'x'.join(str(s) for s in size)]),
                             _swap=not (self.base == 4 and metric == 'euclidean'), # TODO: handle base 4 swapping for the Euclidean metric
                             **xd_kwargs)
                dist = min(dist, func(xd.to(self.device()), **kwargs))
        return dist

    @staticmethod
    def is_architectural(n):

        return not ('weight' in n or n == 'bias')

    def named_arch_params(self):

        return ((n, p) for n, p in self.named_parameters() if self.is_architectural(n))

    def arch_params(self):

        return (p for _, p in self.named_arch_params())

    def named_model_weights(self):

        return ((n, p) for n, p in self.named_parameters() if not self.is_architectural(n))

    def model_weights(self):

        return (p for _, p in self.named_model_weights())
    

if __name__ == '__main__':

    batch, chan, modes, n = 1, 1, 12, 48
    print('Testing XD-ops with FNO warm start')
    errs = 0
    for dims in range(1, 4):
        x = torch.normal(0., 1., (batch, chan, *[n]*dims))
        fno = FNO(chan, chan, [modes]*dims, pad=True)
        true = fno(x)

        for fixed in [False, True]:
            for herm_pad in [False, True]:
                for perm_fix in [False, True]:
                    for low_freq in [False, True]:

                        print('\r\t', dims,
                              ' fixed: ', fixed,
                              ' herm_pad:', herm_pad,
                              ' perm_fix:', perm_fix,
                              ' low_freq:', low_freq,
                              end=' '*10)

                        dK = 4 if low_freq and not perm_fix else 1
                        dL = 2 - (low_freq or perm_fix)
                        dM = 4 if low_freq and not perm_fix else 1
                        xd = XD([n]*dims, chan, chan, arch_init='fno', max_kernel_size=2*modes, global_biasing=False, channel_gating=False, complex_weight=True, kmatrix_depth=(dK, dL, dM), fixed=fixed, herm_pad=herm_pad, perm_fix=perm_fix, low_freq=low_freq)
                        xd.weight.data = torch.zeros(xd.weight.shape, dtype=torch.complex64)
                        for slices in fno.get_slices():
                            xd.weight.data[slices] = torch.flip(fno.weight.data[slices], dims=tuple(range(-dims, 0)))
                        if not herm_pad:
                            xd.weight.data = xd.hermitian_pad(xd.weight.data[[slice(None)]*(dims+1)+[slice(modes)]])
                        err = (torch.norm(xd(x) - true) / torch.norm(true)).item()
                        if err >= 1E-5:
                            errs += 1
                            print('\t comparison', err)
                            pdb.set_trace()
    print('\r\tfound', errs, 'errors' + ' '*50)

    batch, device = 2, 'cuda'
    with torch.no_grad():
        for dims in range(1, 4):
            for name, Op, slc in [
                                  ('Conv', Conv, lambda p: None),
                                  ('Pool', AvgPool, lambda p: [slice(None), slice(None)] + [slice(p, -p) if p else slice(None)]*dims),
                                  ]:
                print('Testing '+str(dims)+'d XD-op with', name, 'warm start')
                errs = 0

                in_size = tuple([32] * dims)
                for biasing in [False, 'additive', 'interp']:
                    for gating in [False, 'complete', 'interp']:
                        kwargs = {'kmatrix_depth': 3, 'global_biasing': biasing, 'channel_gating': gating}
                        for in_channels, out_channels in zip([1, 3 if name == 'Conv' else 16], [1, 16]):
                            if Op == AvgPool and (not biasing or not gating):
                                continue
                            for kernel_size in range(1, 5):
                                for inc in range(3):
                                    for dilation in range(1, 2 + 2*(kernel_size > 1)):
                                        if Op == AvgPool and dilation > 1:
                                            continue
                                        for padding in range(dilation * (kernel_size-1) + 1):
                                            mode = 'zeros' if padding > (kernel_size-1) // 2 else 'circular'
                                            if Op == AvgPool and mode == 'zeros':
                                                continue
                                            for stride in range(1, kernel_size+1):
                                                for size in [in_size[0] // 2, in_size[0] // 2 + 1, in_size[0]-1, in_size[0]]:
                                                    for fixed in [False, True]:
                                                        for fpos in [-1, 0]:
                                                            if fixed and not fpos:
                                                                continue
                                                            print('\r\t', biasing, gating, 
                                                                  ' in:', in_channels, 
                                                                  ' out:', out_channels, 
                                                                  ' kernel:', kernel_size, 
                                                                  ' dilation:', dilation,
                                                                  ' increment:', inc,
                                                                  ' padding:', padding, 
                                                                  ' stride:', stride,
                                                                  ' size:', size,
                                                                  ' fpos:', fpos,
                                                                  ' fixed:', fixed,
                                                                  end=' '*25)

                                                            arch = 'conv_' + 'x'.join([str(kernel_size)]*dims)
                                                            weight = None
                                                            try:
                                                                comp = Op(dims)(in_channels, out_channels, kernel_size, padding=padding, padding_mode=mode if size == in_size[0] else 'zeros', bias=False, stride=stride, dilation=dilation).to(device)
                                                                weight = comp.weight.data
                                                                xd = XD(in_size, in_channels, out_channels, arch_init=arch, max_kernel_size=kernel_size+inc, padding=padding, weight=weight, stride=stride, dilation_init=dilation, padding_mode=mode, fourier_position=fpos, fixed=fixed, **kwargs).to(device)
                                                            except TypeError:
                                                                comp = Op(dims)(kernel_size=kernel_size, padding=padding, stride=stride, count_include_pad=True).to(device)
                                                                arch = arch.replace('conv', 'pool')
                                                                xd = XD(in_size, in_channels, out_channels, arch_init=arch, max_kernel_size=kernel_size+inc, stride=stride, padding=padding, weight=weight, padding_mode=mode, fourier_position=fpos, fixed=fixed, **kwargs).to(device)

                                                            x = torch.normal(0.0, 1.0, (batch, in_channels, *[size]*dims)).to(device)
                                                            true = comp(x)[slc(padding)]
                                                            err = (torch.norm(xd(x)[slc(padding)] - true) / torch.norm(true)).item()
                                                            if err >= 1E-5:
                                                                errs += 1
                                                                print('\t comparison', err)
                                                                pdb.set_trace()

                                    w = xd.weight.clone()
                                    for metric, options in [
                                                            ('euclidean', {}), 
                                                            ('frobenius', {'approx': 1}), 
                                                            ('averaged', {'approx': 1, 'samples': 10}),
                                                            ]:
                                        err = xd.distance_from(src=name.lower(), metric=metric, relative=True, **options)
                                        if err > 1E-16:
                                            errs += 1
                                            print('\t '+metric, err)
                                    err = torch.norm(w - xd.weight) / torch.norm(xd.weight)
                                    if err > 1E-16:
                                        errs += 1
                                        print('\t mismatch', err.item())
                                        pdb.set_trace()

                                    if name == 'Conv':
                                        xd = XD(in_size, in_channels, out_channels, arch_init='ortho', max_kernel_size=kernel_size+inc, weight=weight, **kwargs).to(device)
                                        true = xd.penalty()
                                        err = ((xd.penalty(approx=4**dims) - true) / true).item()
                                        if err >= 0.5 ** dims:
                                            errs += 1
                                            print('\t penalty', err)

                print('\r\tfound', errs, 'errors' + ' '*100)
