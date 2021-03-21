import pdb
import torch
from copy import deepcopy
from torch import autograd, nn, optim
from torch._six import inf
from .utils import int2tuple


class MixedOptimizer(optim.Optimizer):

    def __init__(self, optimizers, alternating=False, op_decay=[], coef=1E-4, approx=16):
        '''
        Args:
            optimizers: list of objects that are subclasses of optim.Optimizer
            alternating: whether to alternate steps with different optimizers
            op_decay: list of objects that are subclasses of nn.Module
            coef: penalty term coefficient for op-decay
            approx: number of basis vectors used to approximate Frobenius norm for op-decay
            model_optimizers: indices of 'optimizers' that are not architecture optimizers
        '''

        self.optimizers = []
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group['method'] = type(optimizer)
                group['initial_lr'] = group.get('initial_lr', group['lr'])
            self.optimizers.append(optimizer)
        super(MixedOptimizer, self).__init__((g for o in self.optimizers for g in o.param_groups), {})
        self.alternating = alternating
        self.iteration = 0
        self.op_decay = op_decay
        self.approx = approx
        self.coef = coef

    def step(self):

        if self.coef and self.op_decay:
            autograd.backward(self.coef * sum(m.penalty(approx=self.approx) for m in self.op_decay))

        if self.alternating:
            self.optimizers[self.iteration % len(self.optimizers)].step()
        else:
            for optimizer in self.optimizers:
                optimizer.step()
        self.iteration += 1


def iter_grad(parameters):

    for param in parameters:
        try:
            yield param.grad.real.detach()
            yield param.grad.imag.detach()
        except RuntimeError:
            yield param.grad.detach()

def clip_grad_norm(parameters, max_norm, norm_type=2.0):
    '''handles gradient clipping for complex parameters'''

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(g.abs().max().to(device) for g in iter_grad(parameters))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g, norm_type).to(device) for g in iter_grad(parameters)]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm
