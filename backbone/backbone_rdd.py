from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        k = 3
        self.conv1 = nn.Conv2d(57, 1, k, 1, padding=(k - 1) // 2)

        #self.conv1 = nn.Conv2d(1, 32, 3, 1)
        #self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #self.dropout1 = nn.Dropout(0.25)
        #self.dropout2 = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(9216, 128)
        #self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        output = F.relu(x)
        return output

class DeepConRddDistances(nn.Module):
    def __init__(self, L=128, num_blocks=8, width=16, expected_n_channels=57,
            no_dilation=False):
        super(DeepConRddDistances, self).__init__()
        self.L = L
        self.num_blocks = num_blocks
        self.width = width
        self.expected_n_channels = expected_n_channels

        self.in_layer = nn.Sequential(
            nn.BatchNorm2d(expected_n_channels), 
            nn.ReLU(),
            nn.Conv2d(expected_n_channels, width, 
                kernel_size=(1, 1), stride=1, padding=0))

        dropout_value = 0.3
        n_channels = width
        d_rate = 1
        dilation = not no_dilation
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            self.blocks.append(nn.Sequential(
                nn.BatchNorm2d(n_channels), 
                nn.ReLU(),
                nn.ZeroPad2d((3-1)//2), ####
                nn.Conv2d(n_channels, n_channels, 
                    kernel_size=(3, 3), stride=1, padding=0),
                nn.Dropout2d(dropout_value),
                nn.ReLU(),
                nn.ZeroPad2d((3-1)*d_rate//2), ####
                nn.Conv2d(n_channels, n_channels, 
                    kernel_size=(3, 3), stride=1, padding=0,
                    dilation=(d_rate, d_rate)),))
            if dilation:
                if d_rate == 1:
                    d_rate = 2
                elif d_rate == 2:
                    d_rate = 4
                else:
                    d_rate = 1

        self.out_layer = nn.Sequential(
            nn.BatchNorm2d(n_channels), 
            nn.ReLU(),
            nn.ZeroPad2d((3-1)//2), ####
            nn.Conv2d(n_channels, 1, 
                kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),)

    def forward(self, x):
        x = x.permute(0,3,1,2).contiguous()
        tower = self.in_layer(x)
        for block in self.blocks:
            b = block(tower)
            tower = b + tower
        output = self.out_layer(tower)
        return output

    def forward_window(self, x, stride=-1):
        L = self.L
        _, _, _, s_length = x.shape

        if stride == -1: # Default to window size
            stride = L
            assert(s_length % L == 0)
        
        y = torch.zeros_like(x)[:, :1, :, :] # TODO Use nans? Use numpy?
        for i in range(0, s_length, stride):
            for j in range(0, s_length, stride):
                out = self.forward(x[:, :, i:i+L, j:j+L])
                y[:, :, i:i+L, j:j+L] = out
        return y
