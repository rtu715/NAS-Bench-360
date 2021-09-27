"""
VGGLike module, as per FSD50k paper [https://arxiv.org/abs/2010.00475]
"""
import torch
from torch import nn


class VGGLike(nn.Module):
    def __init__(self, n_classes):
        super(VGGLike, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, (3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, (3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, (3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, (3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.features(x)
        avg_ = self.avg_pool(x)
        max_ = self.max_pool(x)
        fc_in = torch.cat([avg_, max_], dim=1)
        fc_in = torch.flatten(fc_in, 1)
        fc_in = self.fc(fc_in)
        return fc_in
