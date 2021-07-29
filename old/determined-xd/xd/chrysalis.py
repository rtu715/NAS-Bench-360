import pdb
from copy import deepcopy
import torch
from torch import nn
from .ops import Conv, FNO, get_fno
from .utils import int2tuple
from .xd import XD


def get_module(model, module_string):
    if module_string:
        for substring in module_string.split('.'):
            model = getattr(model, substring)
    return model


WeightNorm = type(list(torch.nn.utils.weight_norm(torch.nn.Conv1d(1, 1, 1))._forward_pre_hooks.items())[0][1])


def check_weight_norm(module):
    for key, value in module._forward_pre_hooks.items():
        if type(value) == WeightNorm:
            return value


class Chrysalis(nn.Sequential):

    @classmethod
    def metamorphosize(cls, model, in_place=False, attrs=[]):
        '''
        Args:
            model: backbone model
            in_place: replace backbone model layers in place
            attrs: custom attributes of model to replace
        '''

        model = model if in_place else deepcopy(model)
        attrs = attrs if attrs else dir(model)
        assert 'forward' in attrs, "if nonempty, 'attrs' must contain 'forward'"
        attrs = [(attr, getattr(model, attr)) for attr in attrs if not attr[:2] == '__']
        model.__class__ = cls
        for name, attr in attrs:
            setattr(model, name, attr)
        return model

    def named_arch_params(self):
        '''iterates through (name, param) pairs of all architecture parameters in the model'''

        for name, module in self.named_modules():
            if name and hasattr(module, 'named_arch_params'):
                for n, p in module.named_arch_params():
                    yield name + '.' + n, p

    def arch_params(self):
        '''iterates through all architecture parameters in the model'''

        return (p for _, p in self.named_arch_params())

    def named_model_weights(self):
        '''iterates through (name, param) pairs of all model weights in the model'''

        exclude = {name for name, _ in self.named_arch_params()}
        return ((n, p) for n, p in self.named_parameters() if not n in exclude)

    def model_weights(self):
        '''iterates through all model weights in the model'''

        return (p for _, p in self.named_model_weights())

    def named_xd_weights(self, include_bias=False):
        '''iterates through (name, param) pairs of all model weights associated with XD objects'''

        for name, module in self.named_modules():
            if name and hasattr(module, 'named_model_weights'):
                for n, p in module.named_model_weights():
                    if include_bias or not n == 'bias':
                        yield name + '.' + n, p

    def xd_weights(self, **kwargs):
        '''iterates through all model weights associated with XD objects'''

        return (p for _, p in self.named_xd_weights(**kwargs))

    def named_nonxd_weights(self, exclude_bias=False):
        '''iterates through (name, param) pairs of all model weights not associated with XD objects'''

        exclude = {name for name, _ in self.named_xd_weights(include_bias=not exclude_bias)}
        return ((n, p) for n, p in self.named_model_weights() if not n in exclude)

    def nonxd_weights(self, **kwargs):
        '''iterates through all model weights not associated with XD objects'''

        return (p for _, p in self.named_nonxd_weights(**kwargs))

    def patch(self, module_string, sample_input, sample_output, test=False, test_boundary=0, func=lambda m: m,
              **kwargs):
        '''patches specified module with a XD
        Args:
            module_string: name of module to replace
            sample_input: sample input into the module.forward function
            sample_output: sample output of the module.forward function
            test: test agreement of replacement module on 'sample_input' and return relative error
            test_boundary: sets boundary when testing replacement module
            func: function to apply to xd.XD object before patching
            kwargs: passed to xd.XD
        '''

        if sample_input is None:
            module, test = None, False
        else:
            in_size = sample_input.shape[2:]
            in_channels = sample_input.shape[1]
            out_channels = sample_output.shape[1]
            module = func(XD(in_size, in_channels, out_channels, **kwargs))

        while True:
            module_split = module_string.split('.')
            parent = get_module(self, '.'.join(module_split[:-1]))
            name = module_split[-1]
            child = getattr(parent, name)
            setattr(parent, module_split[-1], module)
            for module_string, m in self.named_modules():
                if m == child:
                    break
            else:
                break

        if test:
            test_boundary = int2tuple(test_boundary, length=len(in_size))
            slc = [slice(None), slice(None)] + [slice(b, n - b) for b, n in
                                                zip(int2tuple(test_boundary, length=len(in_size)), in_size)]
            output = module(sample_input)
            return module, (torch.norm(output[slc] - sample_output[slc]) / torch.norm(sample_output[slc])).item()
        return module, "module not used in forward pass" if sample_input is None else None

    def collect_io(self, sample_input, modules, *args):

        module_io = {}
        handles = [m.register_forward_hook(lambda s, i, o: module_io.__setitem__(s, (i[0], o))) for m in modules]
        self(sample_input, *args)
        for handle in handles:
            handle.remove()
        return module_io

    def patch_skip(self, sample_input, named_modules=[], warm_start=True, verbose=False, **kwargs):
        '''
        Args:
            sample_input: torch.Tensor of shape [batch-size, input-channels, *input-width]
            named_modules: iterable of named skip-connect modules in self.model
            warm_start: whether to initialize modules as skip-connects
            verbose: print patch logs
            kwargs: passed to self.patch
        '''

        named_modules = list(named_modules)
        module_io = self.collect_io(sample_input, (m for _, m in named_modules))

        for name, module in named_modules:
            skip = 1 + int(type(module) != nn.Sequential)
            m, err = self.patch(name,
                                *module_io[module],
                                test=verbose,
                                arch_init='skip_' + str(skip) if warm_start else 'ortho',
                                stride=skip,
                                **kwargs)
            if verbose:
                print(name, '\terror:', err)

    def patch_pool(self, sample_input, named_modules=[], warm_start=True, verbose=False, **kwargs):
        '''
        Args:
            sample_input: torch.Tensor of shape [batch-size, input-channels, *input-width]
            named_modules: iterable of named skip-connect modules in self.model
            warm_start: whether to initialize modules as skip-connects
            verbose: print patch logs
            kwargs: passed to self.patch
        '''

        named_modules = list(named_modules)
        module_io = self.collect_io(sample_input, (m for _, m in named_modules))

        for name, module in named_modules:
            ks = module.kernel_size
            arch_init = 'pool_' + (str(ks) if type(ks) == int else 'x'.join(str(k) for k in ks))
            m, err = self.patch(name,
                                *module_io[module],
                                test=verbose,
                                test_boundary=1,
                                arch_init=arch_init if warm_start else 'ortho',
                                padding=module.padding,
                                stride=module.stride,
                                **kwargs)
            if verbose:
                print(name, '\terror:', err)

    def patch_conv(self, sample_input, *args, named_modules=None, warm_start=True, verbose=False, kmatrix_depth=1,
                   **kwargs):
        '''
        Args:
            sample_input: torch.Tensor of shape [batch-size, input-channels, *input-width]
            args: additional arguments passed to self.forward
            named_modules: iterable of named modules ; if None uses all modules in self.model
            warm_start: whether to initialize modules as 2d convs
            verbose: print patch logs
            kmatrix_depth: passed to self.patch
            kwargs: passed to self.patch
        '''

        named_modules = self.named_modules() if named_modules is None else named_modules
        named_modules = [(n, m) for n, m in named_modules if
                         hasattr(m, 'kernel_size') and type(m.kernel_size) == tuple and type(m) == Conv(
                             len(m.kernel_size))]
        module_io = self.collect_io(sample_input, (m for _, m in named_modules), *args)
        for name, module in named_modules:
            ks = module.kernel_size
            arch_init = 'conv_' + 'x'.join(str(k) for k in ks)
            wn = check_weight_norm(module)
            msg = ""
            if wn is None:
                func = lambda m: m
            else:
                msg += "\tweight-norm detected"

                def func(m):
                    m = torch.nn.utils.weight_norm(m, dim=wn.dim)
                    if m.weight_g.shape == module.weight_g.shape:
                        m.weight_g = module.weight_g
                        m.weight_v = module.weight_v
                    return m
            depth = int2tuple(kmatrix_depth, length=3)
            if any(d > 1 for d in module.dilation) and depth[1] < 3:
                msg += "\tdepth increased due to dilation"
                depth = (depth[0], 3, depth[2])
            m, err = self.patch(name,
                                *module_io.get(module, (None, None)),
                                test=verbose,
                                test_boundary=1,
                                func=func,
                                arch_init=arch_init if warm_start else 'ortho',
                                kmatrix_depth=depth,
                                padding=module.padding,
                                stride=module.stride,
                                dilation_init=module.dilation,
                                arch_shape=arch_init,
                                weight=module.weight,
                                bias=module.bias,
                                **kwargs)

            if hasattr(m, 'max_kernel_size') and tuple(ks) != m.max_kernel_size:
                msg += "\tnew kernel size: " + str(m.max_kernel_size)
            if verbose:
                print(name, '\terror:', err, msg)

    def patch_fno(self, sample_input, named_modules=[], warm_start=True, verbose=False, kmatrix_depth=1,
                  complex_weight=True, herm_pad=True, perm_fix=True, low_freq=False, **kwargs):
        '''
        Args:
            sample_input: torch.Tensor of shape [batch-size, input-channels, *input-width]
            named_modules: iterable of named modules
            warm_start: whether to initialize modules as 2d convs
            verbose: print patch logs
            kmatrix_depth: passed to self.patch
            complex_weight: passed to self.patch
            herm_pad: passed to self.patch
            perm_fix: passed to self.patch
            low_freq: passed to self.patch
            kwargs: passed to self.patch
        '''

        assert complex_weight, "patching not implemented for FNO with real weights"
        named_modules = list(named_modules)
        module_io = self.collect_io(sample_input, (m for _, m in named_modules))

        for name, module in named_modules:
            if not isinstance(module, FNO):
                fno = get_fno(module)
            else:
                fno = module

            arch_init = 'fno'
            msg = ""
            depth = int2tuple(kmatrix_depth, length=3)
            if low_freq:
                if (depth[0] != depth[2] or min(depth[1], depth[2]) < 4) and not perm_fix:
                    msg += "\tdepth increased for bitreversal"
                    depth = (max(depth[0], depth[2], 4), depth[1], max(depth[0], depth[2], 4))
            elif depth[1] < 2 and not perm_fix:
                msg += "\tdepth increased for bitreversal"
                depth = (depth[0], 2, depth[2])
            weight = fno.weight.clone()
            for slices in fno.get_slices():
                weight[slices] = torch.flip(fno.weight[slices], dims=tuple(range(-fno.dims, 0)))
            if not herm_pad:
                weight = XD.hermitian_pad(weight[[slice(None)] * (fno.dims + 1) + [slice(fno.modes[-1])]])
            try:
                m, err = self.patch(name,
                                    *module_io.get(module, (None, None)),
                                    test=verbose,
                                    arch_init='fno' if warm_start else 'ortho',
                                    kmatrix_depth=depth,
                                    weight=weight,
                                    max_kernel_size=[2 * m for m in fno.modes],
                                    complex_weight=complex_weight,
                                    herm_pad=herm_pad,
                                    perm_fix=perm_fix,
                                    low_freq=low_freq,
                                    **kwargs)
            except:
                pdb.set_trace()
            if verbose:
                print(name, '\terror:', err, msg)

    def save_arch(self, path):
        '''saves architecture parameters to provided filepath'''

        torch.save(dict(self.named_arch_params()), path)

    def load_arch(self, path, verbose=False):
        '''loads architecture parameters from provided filepath'''

        data = torch.load(path)
        for n, p in self.named_arch_params():
            load = data[n].data
            if p.data.shape == load.shape:
                p.data = load.to(p.device)
            elif verbose:
                print('did not load', n, '(shape mismatch)')

    def set_arch_requires_grad(self, requires_grad):
        '''sets 'requires_grad' attribute of architecture parameters to given value'''

        for param in self.arch_params():
            param.requires_grad = bool(requires_grad)


if __name__ == '__main__':

    import sys;

    '''
    sys.path.insert(0, './fourier_neural_operator')
    from collections import OrderedDict
    from fourier_1d import SpectralConv1d
    from fourier_2d import SpectralConv2d
    from fourier_3d import SpectralConv3d_fast
    from ops import AvgPool

    chan, modes = 2, 12
    for dims in range(1, 4):
        for herm_pad in [False, True]:
            for perm_fix in [False, True]:
                for low_freq in [False, True]:
                    module = {1: SpectralConv1d,
                              2: SpectralConv2d,
                              3: SpectralConv3d_fast}[dims]
                    model = nn.Sequential(OrderedDict([('sconv1', module(chan, chan, *[modes] * dims)),
                                                       ('pool', AvgPool(dims)(3)),
                                                       ('sconv2', module(chan, chan, *[modes] * dims))]))
                    model = Chrysalis.metamorphosize(model)
                    model.patch_fno(torch.normal(0., 1., (2, chan, *[96] * dims)),
                                    named_modules=[('sconv1', model.sconv1),
                                                   ('sconv2', model.sconv2)],
                                    verbose=True,
                                    herm_pad=herm_pad,
                                    perm_fix=perm_fix,
                                    low_freq=low_freq)
    '''
