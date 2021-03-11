import pdb
from copy import deepcopy
from operator import itemgetter
import torch
from torch import nn
from torch.nn import functional as F
from chrysalis import Chrysalis, get_module
from ops import Conv
from utils import int2tuple
from xd import XD


OPERATIONS = [
              ('conv_3', {'kernel_size': 3, 'padding': 1}),
              ('conv_5', {'kernel_size': 5, 'padding': 2}),
              ('dil-conv_3', {'kernel_size': 3, 'padding': 2, 'dilation': 2}),
              ('dil-conv_5', {'kernel_size': 5, 'padding': 4, 'dilation': 2}),
              ('max-pool_3', {'kernel_size': 3, 'padding': 1}),
              ('avg-pool_3', {'kernel_size': 3, 'padding': 1}),
              ('skip', {}),
              ('zero', {}),
              ]


def MaxPool(dims):

    return getattr(nn, 'MaxPool'+str(dims)+'d')


class ZeroOp(nn.Module):

    @staticmethod
    def _offset_select(kernel, sizes):

        offsets = [(o-k+int(o % 2 == 1)) // 2 for o, k in zip(kernel.shape[-len(sizes):], sizes)]
        slices = [slice(None)] * (len(kernel.shape) - len(sizes))
        return kernel[slices + [slice(offset, offset+size) for offset, size in zip(offsets, sizes)]]

    @staticmethod
    def _offset_pad(output, sizes):

        pad = []
        for i in range(1, len(sizes)+1):
            diff = max(0, sizes[-i] - output.shape[-i])
            pad.append(diff - diff // 2)
            pad.append(diff // 2)
        return F.pad(output, pad)

    def _offset_adjust(self, input):

        return self._offset_select(self._offset_pad(input, self.out_size), self.out_size)

    def __init__(self, out_size, out_channels):

        super(ZeroOp, self).__init__()
        self.out_size = [out_channels] + list(out_size)

    def forward(self, input):

        return self._offset_adjust(input.mul(0.))


class SkipConnect(ZeroOp):

    def __init__(self, out_size, out_channels, stride=1):

        super(SkipConnect, self).__init__(out_size, out_channels)
        self.stride = [slice(None)]*2 + [slice(None, None, s) for s in int2tuple(stride, length=len(out_size))]

    def forward(self, input):

        return self._offset_adjust(input[self.stride])


class PoolNd(ZeroOp):

    def __init__(self, out_size, pool, out_channels, **kwargs):

        super(PoolNd, self).__init__(out_size, out_channels)
        self.func = getattr(F, pool+'_pool'+str(len(out_size))+'d')
        self.kwargs = kwargs

    def forward(self, input):

        output = self.func(input, **self.kwargs)
        return self._offset_adjust(output)


class ConvNd(ZeroOp):

    def __init__(self, out_size, in_channels, out_channels, **kwargs):

        super(ConvNd, self).__init__(out_size, out_channels)
        conv = Conv(len(out_size))(in_channels, out_channels, **kwargs)
        self.weight = conv.weight
        self.weight_size = self.weight.shape
        self.kwargs = {k: v for k, v in kwargs.items() if not k == 'kernel_size'}
        self.func = getattr(F, 'conv'+str(len(out_size))+'d')

    def forward(self, input):

        weight = self._offset_select(self.weight, self.weight_size)
        output = self.func(input, weight, **self.kwargs)
        return self._offset_adjust(output)


class DARTS(nn.Module):

    def __init__(self, out_size, in_channels, out_channels, operations=OPERATIONS, arch_init='uniform', weight_init=nn.init.kaiming_normal_, weight=None, perturb=0.1, bias=None, **kwargs):

        super(DARTS, self).__init__()

        self.dims = 2 if type(out_size) == int else len(out_size)
        self.out_size = int2tuple(out_size, length=self.dims)
        self.zeroop = ZeroOp(self.out_size, out_channels)
        self.operations = nn.ModuleDict()
        self.logits = nn.ParameterDict()
        weight_size = [out_channels, in_channels] + [0] * self.dims
        for name, op_kwargs in operations:
            op_kwargs = deepcopy(op_kwargs)
            op_kwargs.update(kwargs)
            if 'conv' in name:
                self.operations[name] = ConvNd(self.out_size, in_channels, out_channels, **op_kwargs)
                for i, size in enumerate(self.operations[name].weight.shape):
                    weight_size[i] = max(weight_size[i], size)
            elif name[:4] == 'skip':
                self.operations[name] = SkipConnect(self.out_size, out_channels, **op_kwargs)
            elif name[:4] == 'zero':
                self.operations[name] = self.zeroop
            else:
                self.operations[name] = PoolNd(out_size, name[:3], out_channels, **op_kwargs)
            if arch_init == 'uniform':
                p = 1.
            elif arch_init == name:
                p = (1.-perturb) * len(operations)
            else:
                p = perturb * len(operations) /  (len(operations) - 1)
            self.logits[name] = nn.Parameter(torch.log(torch.Tensor([p])))

        self.weight = nn.Parameter(torch.empty(weight_size))
        weight_init(self.weight)
        if not weight is None:
            XD._offset_insert(self.weight.data, weight)
        for name, module in self.operations.items():
            if hasattr(module, 'weight'):
                module.weight = self.weight

        self.bias = nn.Parameter(bias) if type(bias) == torch.Tensor else bias
        self.discrete = ''

    def discretize(self):

        self.discrete = max(self.logits.items(), key=itemgetter(1))[0]

    def forward(self, input):

        if self.discrete:
            output = self.operations[self.discrete](input)

        else:
            sumexplogit = sum(torch.exp(logit) for logit in self.logits.values())
            output = self.zeroop(input)
            for name, operation in self.operations.items():
                p = torch.exp(self.logits[name]) / sumexplogit
                output += p * operation(input)

        if self.bias is None:
            return output
        return output + self.bias.reshape(1, *self.bias.shape, *[1]*self.dims)

    def named_arch_params(self):

        return ((n, p) for n, p in self.named_parameters() if 'logits' in n)

    def arch_params(self):

        return (p for _, p in self.named_arch_params())

    def named_model_weights(self):

        return iter([('weight', self.weight)])

    def model_weights(self):

        return iter([self.weight])


class Supernet(Chrysalis):

    def patch_darts(self, sample_input, *args, named_modules=[], warm_start=True, verbose=False, pool_patch='max-pool_3', **kwargs):

        named_modules = list(named_modules)
        module_io = self.collect_io(sample_input, (m for _, m in named_modules))
        for name, module in named_modules:

            op_kwargs = deepcopy(kwargs)
            if warm_start:
                try:
                    arch_init = 'conv_' + str(module.kernel_size[0])
                    op_kwargs['stride'] = module.stride
                except AttributeError:
                    arch_init = 'skip'
                except TypeError:
                    arch_init = pool_patch
            else:
                arch_init = 'uniform'
            weight = getattr(module, 'weight', None)

            inp, out = module_io.get(module, (None, None))
            if inp is None:
                if verbose:
                    print(name, '\terror:', 'module not used in forward pass')
                continue
            in_channels = inp.shape[1]
            out_channels = out.shape[1]
            if arch_init == 'skip':
                op_kwargs['stride'] = inp.shape[-1] // out.shape[-1]
            out_size = out.shape[2:]
            mod = DARTS(out_size,
                        in_channels,
                        out_channels,
                        arch_init=arch_init,
                        bias=getattr(module, 'bias', None),
                        weight=weight, **op_kwargs)

            module_string = name
            while True:
                module_split = module_string.split('.')
                parent = get_module(self, '.'.join(module_split[:-1]))
                nam = module_split[-1]
                child = getattr(parent, nam)
                setattr(parent, module_split[-1], mod)
                for module_string, m in self.named_modules():
                    if m == child:
                        break
                else:
                    break

            if verbose:
                output = mod(inp)
                err = (torch.norm(output-out) / torch.norm(out)).item()
                print(name, '\terror:', err)

    def discretize(self):

        for module in self.modules():
            if type(module) == DARTS:
                module.discretize()
