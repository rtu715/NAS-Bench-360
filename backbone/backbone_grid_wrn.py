"""
Reference: https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
Implementation of Wide Resnet as NAS backbone architecture

Parameter count 28-10: 36.5M
Parameter count for 40-4: 8.9M
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    '''
    first applies batch norm and relu before applying convolution
    we can change the order of operations if needed
    '''

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class Backbone(nn.Module):
    '''
    wide resnet 50
    '''
    def __init__(self, depth, num_classes, widen_factor=2, in_channels=3, dropRate=0.0):
        super(Backbone, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(in_channels, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.in_layer = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                self.conv1)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 1, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 1, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.out_conv = nn.Conv2d(nChannels[3], num_classes, kernel_size=3, stride=1, padding=1)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        pde = False
        if x.size(3) == 3:
            pde = True
            #pde input needs to be permuted
            x = x.permute(0,3,1,2).contiguous()
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        #if pde:
        #    self.pool= nn.AdaptiveAvgPool2d(85)
        #else:
        #    self.pool = nn.AdaptiveAvgPool2d(128)
        #out = self.pool(out)
        #return self.out_conv(out)
        return self.fc(out.permute(0,2,3,1).contiguous())

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
                out = out.permute(0,3,1,2).contiguous()
                y[:, :, ip:ip + L, jp:jp + L] += out
                counts[:, :, ip:ip + L, jp:jp + L] += torch.ones_like(out)
        return y / counts
