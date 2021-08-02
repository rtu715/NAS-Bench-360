import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .operations import OPS
from tools.utils import parse_net_config


class Block(nn.Module):

    def __init__(self, in_ch, block_ch, head_op, stack_ops, stride):
        super(Block, self).__init__()
        self.head_layer = OPS[head_op](in_ch, block_ch, stride, 
                                        affine=True, track_running_stats=True)

        modules = []
        for stack_op in stack_ops:
            modules.append(OPS[stack_op](block_ch, block_ch, 1, 
                                        affine=True, track_running_stats=True))
        self.stack_layers = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.head_layer(x)
        x = self.stack_layers(x)
        return x


class Conv1_1_Block(nn.Module):

    def __init__(self, in_ch, block_ch):
        super(Conv1_1_Block, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=block_ch,
                        kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(block_ch),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv1_1(x)



class RES_Net(nn.Module):
    def __init__(self, net_config, task='cifar10', config=None):
        """
        net_config=[[in_ch, out_ch], head_op, [stack_ops], num_stack_layers, stride]
        """
        super(RES_Net, self).__init__()
        self.config = config
        self.net_config = parse_net_config(net_config)
        self.in_chs = self.net_config[0][0][0]
        self.dataset = task
        dataset_hypers = {'ECG':(4,1), 'satellite':(24,1)}

        n_classes, in_channels = dataset_hypers[self.dataset]
        self._num_classes = n_classes
        #input stride changed to 1 from 2
        self.input_block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=self.in_chs, kernel_size=3,
                    stride=1, padding=1, bias=False),
            nn.BatchNorm1d(self.in_chs),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.ModuleList()
        for config in self.net_config:
            self.blocks.append(Block(config[0][0], config[0][1], 
                            config[1], config[2], config[-1]))

        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        if self.net_config[-1][1] == 'bottle_neck':
            last_dim = self.net_config[-1][0][-1] * 4
        else:
            last_dim = self.net_config[-1][0][1]
        self.classifier = nn.Linear(last_dim, self._num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                if m.affine==True:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):        
        block_data = self.input_block(x)
        for i, block in enumerate(self.blocks):
            block_data = block(block_data)

        out = self.global_pooling(block_data)
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits
