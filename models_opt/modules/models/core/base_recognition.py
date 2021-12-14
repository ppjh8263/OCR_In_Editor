# import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, inputs):
        recurrent, _ = self.rnn(inputs)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.conv = nn.Conv2d(nIn, nOut, 3, 1, 1)
        self.bn = nn.BatchNorm2d(nOut)
        self.relu = nn.ReLU(True)
    def forward(self, inputs):
        conv = self.conv(inputs)
        bn = self.bn(conv)
        return self.relu(bn)

class BaseRecognition(nn.Module):
    def __init__(self, nc, nclass, nh, leakyRelu=False):
        super().__init__()

        self.cnn = nn.Sequential(
            ConvBNReLU(nc, 64),
            ConvBNReLU(64, 64),
            nn.MaxPool2d((2, 1), (2, 1)),
            ConvBNReLU(64, 128),
            ConvBNReLU(128, 128),
            nn.MaxPool2d((2, 1), (2, 1)),
            ConvBNReLU(128, 256),
            ConvBNReLU(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),
            # Trial #006
            # ConvBNReLU(256, 512),
            # nn.MaxPool2d((2, 1), (2, 1))
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(256, nh, nclass)
            # Trial #006
            # BidirectionalLSTM(512, nh, nclass)
        )

    def forward(self, inputs):
        # conv features
        conv = self.cnn(inputs)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        # add log_softmax to converge output
        output = F.log_softmax(output, dim=2)

        return output