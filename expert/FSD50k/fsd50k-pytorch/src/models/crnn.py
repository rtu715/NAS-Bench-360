import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class ConvReLUBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding_size=2):
        super(ConvReLUBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding_size)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        o = self.conv(x)
        o = self.relu(o)
        o = self.bn(o)
        # print(o.shape)
        return o


class CRNN(nn.Module):
    """
    CRNN model, as described in FSD50k paper Sec 5.B.2
    """
    def __init__(self, imgH=96, num_classes=200, nh=64):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.kernel_sizes = [5, 5, 5]
        self.padding_sizes = [1, 1, 1]
        self.strides = [1, 1, 1]
        self.num_filters = [128, 128, 128]

        self.mp_sizes = [(5, 2), (4, 2), (2, 2)]        # in paper, temporal and frequency axis are flipped

        self.cnn = nn.Sequential()
        for ix in range(len(self.kernel_sizes)):
            self._add_block(ix)
        self.rnn = BidirectionalLSTM(128, nh, num_classes)

    def _add_block(self, ix):
        in_channels = 1 if ix == 0 else self.num_filters[ix - 1]
        out_channels = self.num_filters[ix]
        self.cnn.add_module("cnn_block{}".format(ix), ConvReLUBN(in_channels, out_channels, self.kernel_sizes[ix],
                                                                 self.strides[ix], self.padding_sizes[ix]))
        self.cnn.add_module("pooling{}".format(ix), nn.MaxPool2d(self.mp_sizes[ix]))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        # print(b, c, h, w)
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        output = output[-1]     # take the last sequence as output
        return output
