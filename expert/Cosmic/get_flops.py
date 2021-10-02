import torch
from ptflops import get_model_complexity_info

from torch import sigmoid
from deepCR.parts import *


class UNet2Sigmoid(nn.Module):
    def __init__(self, n_channels, n_classes, hidden=32):
        super(type(self), self).__init__()
        self.inc = inconv(n_channels, hidden)
        self.down1 = down(hidden, hidden * 2)
        self.up8 = up(hidden * 2, hidden)
        self.outc = outconv(hidden, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up8(x2, x1)
        x = self.outc(x)
        return sigmoid(x)

model = UNet2Sigmoid(1,1,32)
#model = torch.load('2021-09-10_mymodels2_epoch50.pth')
macs, params = get_model_complexity_info(model, (1, 128, 128), as_strings=True,
                                       print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
