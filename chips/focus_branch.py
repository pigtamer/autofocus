import matplotlib.pyplot as plt
import mxnet as mx
from symbols import resnet50
from mxnet.gluon import nn
from mxnet import nd,sym


def genChip(x):
    # TODO:
    #   implement post-process of focus chip generation
    return 0


class FocusBranch(nn.Block):
    def __init__(self, **kwargs):
        super(FocusBranch, self).__init__()
        self.net = nn.Sequential()
        self.net.add(
            nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1),
            nn.Activation(activation='relu'),
            nn.Conv2D(channels=256, kernel_size=1, strides=1, padding=0),
            nn.Activation(activation='relu'),
            nn.Conv2D(channels=256, kernel_size=1, strides=1, padding=0),
            nn.Activation(activation='relu'),
        )

    def forward(self, x, **kwargs):
        x = self.net(x)
        x = nd.softmax(x, axis=1)
        # res = genChip(x)
        return x


class HybridFocusBranch(nn.HybridBlock):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = nn.HybridSequential()
        self.net.add(
            nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1),
            nn.Activation(activation='relu'),
            nn.Conv2D(channels=256, kernel_size=1, strides=1, padding=0),
            nn.Activation(activation='relu'),
            nn.Conv2D(channels=256, kernel_size=1, strides=1, padding=0),
            nn.Activation(activation='relu'),
        )

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.net(x)
        x = sym.softmax(x, axis=1)
        # res = genChip(x)
        return x

def test():
    x = nd.random.normal(0, 1, (1, 3, 512, 512), ctx = mx.gpu())
    basenet = resnet50.ResNet50(params=resnet50.params, IF_DENSE=False)
    basenet.initialize(ctx = mx.gpu())
    x = basenet(x)
    focusnet = FocusBranch()
    focusnet.initialize(ctx = mx.gpu())
    x = focusnet(x)
    print(x)