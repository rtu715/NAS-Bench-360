import torch
import torch.nn as nn

OPS = {
    "none": lambda C, stride, affine, activation_function: Zero(stride),
    "avg_pool_3x3": lambda C, stride, affine, activation_function: nn.AvgPool1d(
        3, stride=stride, padding=1, count_include_pad=False
    ),
    "max_pool_3x3": lambda C, stride, affine, activation_function: nn.MaxPool1d(
        3, stride=stride, padding=1
    ),
    "skip_connect": lambda C, stride, affine, activation_function: Identity()
    if stride == 1
    else FactorizedReduce(C, C, affine=affine),
    "sep_conv_3x3": lambda C, stride, affine, activation_function: SepConv(
        C, C, 3, stride, 1, affine=affine, activation_function=activation_function
    ),
    "sep_conv_5x5": lambda C, stride, affine, activation_function: SepConv(
        C, C, 5, stride, 2, affine=affine, activation_function=activation_function
    ),
    "sep_conv_7x7": lambda C, stride, affine, activation_function: SepConv(
        C, C, 7, stride, 3, affine=affine, activation_function=activation_function
    ),
    "dil_conv_3x3": lambda C, stride, affine, activation_function: DilConv(
        C, C, 3, stride, 2, 2, affine=affine, activation_function=activation_function
    ),
    "dil_conv_5x5": lambda C, stride, affine, activation_function: DilConv(
        C, C, 5, stride, 4, 2, affine=affine, activation_function=activation_function
    ),
}


class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(
                C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm1d(C_out, affine=affine, momentum=0.999, eps=0.001),
        )

    def forward(self, x):
        return self.op(x)


class ActivationConvBN(nn.Module):
    def __init__(
        self,
        activation_function,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        affine=True,
    ):
        super(ActivationConvBN, self).__init__()
        self.op = nn.Sequential(
            activation_function(),
            nn.Conv1d(
                C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm1d(C_out, affine=affine, momentum=0.999, eps=0.001),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine=True,
        activation_function=nn.ReLU,
    ):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            activation_function(),
            nn.Conv1d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=C_in,
                bias=False,
            ),
            nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(C_out, affine=affine, momentum=0.999, eps=0.001),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        affine=True,
        activation_function=nn.ReLU,
    ):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            activation_function(),
            nn.Conv1d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv1d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(C_in, affine=affine, momentum=0.999, eps=0.001),
            activation_function(),
            nn.Conv1d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv1d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(C_out, affine=affine, momentum=0.999, eps=0.001),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True, activation_function=nn.ReLU):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.activation = activation_function()
        self.conv_1 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(C_out, affine=affine, momentum=0.999, eps=0.001)

    def forward(self, x):
        x = self.activation(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x)], dim=1)
        out = self.bn(out)
        return out
