import matplotlib.pyplot as plt
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd

params = {
    "channels":
        ([[64, 64, 256]] * 3,
         [[128, 128, 512]] * 4,
         [[256, 256, 1024]] * 6,
         [[512, 512, 2048]] * 3),

    "ksizes":
        ([[1, 3, 1]] * 3,
         [[1, 3, 1]] * 4,
         [[1, 3, 1]] * 6,
         [[1, 3, 1]] * 3),

    "branches":
        ({"channel": 256, "ksize": 1, "stride": 1, "padding": 0},
         {"channel": 512, "ksize": 1, "stride": 2, "padding": 0},
         {"channel": 1024, "ksize": 1, "stride": 2, "padding": 0},
         {"channel": 2048, "ksize": 1, "stride": 2, "padding": 0})
}


class ResidualUnit(nn.Block):
    def __init__(self, l_chans, l_ksizes, chan_params={None}, bbn=False, resize=False):
        super(ResidualUnit, self).__init__()
        """
        :param l_chans: list of channels for layers on trunk
        :param l_ksizes: list of kernel sizes for layers on trunk
        :param l_strides: list of strides for layers on trunk
        :param chan_params: dict!! params for branch(shortcut) if needed.
        :param bbn: bool. switch for using shortcut in this unit.
        """
        assert len(l_ksizes) == len(l_chans)
        self.bbn = bbn
        self.branch = nn.Sequential()
        if bbn or self.bbn:
            self.branch.add(
                nn.Conv2D(channels=chan_params["channel"], kernel_size=chan_params["ksize"],
                          strides=chan_params["stride"], padding=chan_params["padding"]),
                nn.Activation('relu'),
                nn.BatchNorm(in_channels=chan_params["channel"])
            )
        else:
            self.branch.add(
                nn.Conv2D(channels=l_chans[-1], kernel_size=1,
                          strides=1),
                nn.Activation('relu'),
                nn.BatchNorm(in_channels=l_chans[-1])
            )

        self.trunk = nn.Sequential()
        for k in range(len(l_chans)):
            self.trunk.add(
                nn.Conv2D(channels=l_chans[k], kernel_size=l_ksizes[k],
                          strides=1, padding=int(l_ksizes[k] / 2)),
                nn.Activation(activation='relu'),
                nn.BatchNorm(in_channels=l_chans[k])
            )

        if resize:
            self.trunk[0]._kwargs["stride"] = (2, 2)
            self.branch[0]._kwargs["stride"] = (2, 2)

        self.final_relu = nn.Activation(activation='relu')

    def forward(self, x, *args):
        res_shortcut = self.branch(x)
        res_trunk = self.trunk(x)
        res = self.final_relu(res_shortcut + res_trunk)
        # print(res.shape)
        return res


class ResidualLayer(nn.Block):
    def __init__(self, num_stages, ll_chans, ll_ksizes, chan_param_onlybranch, **kwargs):
        super(ResidualLayer, self).__init__(**kwargs)
        """
        :param num_stages: int, 
            number of residual units in this residual layer
        :param ll_chans: list/tuple (of list), 
            contains channel numbers for sublayers in each residual layer
        :param ll_ksizes: list/tuple (of list), 
            contains kernel sizes for sublayers in each residual layer
        :param chan_param_onlybranch: list/tuple, 
            the param for the branch of this residual layer
        :param kwargs: ~
        """
        self.trunk = nn.Sequential()
        for k in range(num_stages):
            if k == 0:
                self.trunk.add(ResidualUnit(ll_chans[k], ll_ksizes[k],
                                            chan_param_onlybranch, bbn=True, resize=True))
            else:
                self.trunk.add(ResidualUnit(ll_chans[k], ll_ksizes[k]))

    def forward(self, x, *args):
        return self.trunk(x)


class ResNet50(nn.Block):
    def __init__(self, params, **kwargs):
        super(ResNet50, self).__init__(**kwargs)
        chans = params["channels"]
        ksizes = params["ksizes"]
        branches = params["branches"]
        assert len(chans) == len(ksizes) & len(ksizes) == len(branches)
        self.net = nn.Sequential()
        conv1 = nn.Sequential()
        conv1.add(
            nn.Conv2D(channels=64, kernel_size=7,
                      strides=2, padding=int(7 / 2)),
            nn.Activation(activation='relu'),
            nn.BatchNorm(in_channels=64)
        )
        self.net.add(conv1)
        for k in range(len(chans)):
            self.net.add(
                ResidualLayer(len(chans[k]), chans[k], ksizes[k], branches[k])
            )
        self.net.add(
            nn.MaxPool2D(),
            nn.Dense(1000)
        )

    def forward(self, x, *args):
        return self.net(x)


def test():
    import time
    ctx = mx.gpu()
    x = nd.random.normal(0, 1, (1, 3, 224, 224), ctx=ctx)
    net = ResidualUnit([128, 128, 128], [3, 3, 3],
                       chan_params={"channel": 128, "ksize": 1, "stride": 1, "padding": 0},
                       bbn=False, resize=True)
    # print(net)
    len(params["channels"][0])
    net = ResidualLayer(3, params["channels"][0], params["ksizes"][0],
                        params["branches"][0])
    net = ResNet50(params)
    net.initialize(ctx=ctx)
    cnt = time.time()
    net(x)
    print(time.time() - cnt)
    # print(net)
