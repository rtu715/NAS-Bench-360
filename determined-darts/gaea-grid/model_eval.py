from functools import partial

import torch
import torch.nn as nn
from torch.autograd import Variable

from operations import *
from utils import drop_path


class Cell(nn.Module):
    def __init__(
        self,
        genotype,
        C_prev_prev,
        C_prev,
        C,
        reduction,
        reduction_prev,
        activation_function=nn.ReLU,
        drop_prob=0,
    ):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        #self.drop_prob = drop_prob
        #self.drop_module = nn.Dropout2d(drop_prob)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ActivationConvBN(
                activation_function, C_prev_prev, C, 1, 1, 0
            )
        self.preprocess1 = ActivationConvBN(activation_function, C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction, activation_function)

    def _compile(self, C, op_names, indices, concat, reduction, activation_function):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            #if "conv" in name and self.drop_prob > 0:
            #    op = nn.Sequential(self.drop_module, op)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_path_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_path_prob > 0.0:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_path_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_path_prob)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)



class DiscretizedNetwork(nn.Module):
    def __init__(self, C, num_classes, layers, genotype, in_channels, drop_path_prob):
        super(DiscretizedNetwork, self).__init__()
        self._layers = layers
        self.drop_path_prob = drop_path_prob

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                #C_curr *= 2
                #reduction = True
                pass
            else:
                reduction = False
            cell = Cell(
                genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            print('cell multiplier: ',cell.multiplier)

        self.classifier = nn.Linear(C_prev, num_classes)

    def get_save_states(self):
        return {
            "state_dict": self.state_dict(),
        }

    def load_states(self, save_states):
        self.load_state_dict(save_states["state_dict"])

    def forward(self, input, **kwargs):
        s0 = s1 = self.stem(input.permute(0,3,1,2).contiguous())
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

        #out = self.global_pooling(s1)
        out = s1
        logits = self.classifier(out.permute(0,2,3,1).contiguous())
        return logits


    def forward_window(self, x, L, stride=-1):
        _, _, _, s_length = x.shape

        if stride == -1:  # Default to window size
            stride = L
            assert (s_length % L == 0)

        y = torch.zeros_like(x)[:, :1, :, :]
        counts = torch.zeros_like(x)[:, :1, :, :]
        for i in range((((s_length - L) // stride)) + 1):
            ip = i * stride
            for j in range((((s_length - L) // stride)) + 1):
                jp = j * stride
                out = self.forward(x[:, :, ip:ip + L, jp:jp + L])
                out = out.permute(0, 3, 1, 2)
                y[:, :, ip:ip + L, jp:jp + L] += out
                counts[:, :, ip:ip + L, jp:jp + L] += torch.ones_like(out)
        return y / counts
