from collections import namedtuple
from typing import Any

import os
import gzip
import pickle

import numpy as np
import torch
from torch import nn

import torch.utils.data as data_utils

import torchvision
from torchvision import transforms

# From: https://github.com/quark0/DARTS
Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.cur = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class EMA(nn.Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu

    def register(self, params):
        # We register copied tensors to buffer so they will
        # be saved as part of state_dict.
        for i, p in enumerate(params):
            copy = p.clone().detach()
            self.register_buffer("shadow" + str(i), copy)

    def shadow_vars(self):
        for b in self.buffers():
            yield b

    def forward(self, new_params):
        for avg, new in zip(self.shadow_vars(), new_params):
            new_avg = self.mu * avg + (1 - self.mu) * new.detach()
            avg.data = new_avg.data


class EMAWrapper(nn.Module):
    def __init__(self, ema_decay, model):
        super(EMAWrapper, self).__init__()
        self.model = model
        self.ema = EMA(ema_decay)
        self.ema.register(self.ema_vars())

        # Create copies in case we have to resume.
        for i, p in enumerate(self.ema_vars()):
            copy = p.clone().detach()
            self.register_buffer("curr" + str(i), copy)

    def curr_vars(self):
        for n, b in self.named_buffers():
            if n[0:4] == "curr":
                yield b

    def ema_vars(self):
        for p in self.model.parameters():
            yield p
        for n, b in self.model.named_buffers():
            if "running_mean" or "running_var" in n:
                yield b

    def forward(self, *args):
        return self.model(*args)

    def update_ema(self):
        self.ema(self.ema_vars())

    def restore_ema(self):
        for curr, shad, p in zip(
            self.curr_vars(), self.ema.shadow_vars(), self.ema_vars()
        ):
            curr.data = p.data
            p.data = shad.data

    def restore_latest(self):
        for curr, p in zip(self.curr_vars(), self.ema_vars()):
            p.data = curr.data


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


# From: https://github.com/yuhuixu1993/PC-DARTS
class CrossEntropyLabelSmooth(nn.Module):
    """
    Assign small probability to non-target classes to hopefully learn faster and more generalizable features.

    See this paper for more info:
    https://arxiv.org/pdf/1906.02629.pdf
    """

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


