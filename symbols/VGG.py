import numpy as np
import matplotlib.pyplot as plt
from mxnet import nd, sym, gluon
from mxnet.gluon import nn

class Hybrid_BaseNetwork(nn.HybridBlock):  # VGG base network, without fc
    def __init__(self, IF_TINY=True, **kwargs):
        super(Hybrid_BaseNetwork, self).__init__(**kwargs)
        self.IF_TINY = IF_TINY
        self.conv1_1 = nn.Conv2D(channels=64, kernel_size=3, padding=1, activation='relu')
        self.conv1_2 = nn.Conv2D(channels=64, kernel_size=3, padding=1, activation='relu')
        self.pool1 = nn.MaxPool2D(pool_size=(2, 2))
        self.conv2_1 = nn.Conv2D(channels=128, kernel_size=3, padding=1, activation='relu')
        self.conv2_2 = nn.Conv2D(channels=128, kernel_size=3, padding=1, activation='relu')
        self.pool2 = nn.MaxPool2D(pool_size=(2, 2))
        self.conv3_1 = nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu')
        self.conv3_2 = nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu')
        self.conv3_3 = nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu')
        self.pool3 = nn.MaxPool2D(pool_size=(2, 2))  # smaller here
        if not self.IF_TINY:
            self.conv4_1 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.conv4_2 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.conv4_3 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.pool4 = nn.MaxPool2D(pool_size=(2, 2))
            self.conv5_1 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.conv5_2 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.conv5_3 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.pool5 = nn.MaxPool2D(pool_size=(2, 2))

    def hybrid_forward(self, F, x):
        x = self.pool1(self.conv1_2(self.conv1_1(x)))
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        x = self.pool3(self.conv3_3(self.conv3_2(self.conv3_1(x))))
        if not self.IF_TINY:
            x = self.pool4(self.conv4_3(self.conv4_2(self.conv4_1(x))))
            x = self.pool5(self.conv5_3(self.conv5_2(self.conv5_1(x))))
        return x

class BaseNetwork(nn.Block):  # VGG base network, without fc
    def __init__(self, IF_TINY=True, **kwargs):
        super(BaseNetwork, self).__init__(**kwargs)
        self.IF_TINY = IF_TINY
        self.conv1_1 = nn.Conv2D(channels=64, kernel_size=3, padding=1, activation='relu')
        self.conv1_2 = nn.Conv2D(channels=64, kernel_size=3, padding=1, activation='relu')
        self.pool1 = nn.MaxPool2D(pool_size=(2, 2))
        self.conv2_1 = nn.Conv2D(channels=128, kernel_size=3, padding=1, activation='relu')
        self.conv2_2 = nn.Conv2D(channels=128, kernel_size=3, padding=1, activation='relu')
        self.pool2 = nn.MaxPool2D(pool_size=(2, 2))
        self.conv3_1 = nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu')
        self.conv3_2 = nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu')
        self.conv3_3 = nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu')
        self.pool3 = nn.MaxPool2D(pool_size=(2, 2))  # smaller here
        if not self.IF_TINY:
            self.conv4_1 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.conv4_2 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.conv4_3 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.pool4 = nn.MaxPool2D(pool_size=(2, 2))
            self.conv5_1 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.conv5_2 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.conv5_3 = nn.Conv2D(channels=512, kernel_size=3, padding=1, activation='relu')
            self.pool5 = nn.MaxPool2D(pool_size=(2, 2))

    def forward(self, x):
        x = self.pool1(self.conv1_2(self.conv1_1(x)))
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        x = self.pool3(self.conv3_3(self.conv3_2(self.conv3_1(x))))
        if not self.IF_TINY:
            x = self.pool4(self.conv4_3(self.conv4_2(self.conv4_1(x))))
            x = self.pool5(self.conv5_3(self.conv5_2(self.conv5_1(x))))
        return x