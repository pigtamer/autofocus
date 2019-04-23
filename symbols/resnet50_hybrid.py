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

    "kernel_sizes":

        ([[1, 3, 1]] * 3,
         [[1, 3, 1]] * 4,
         [[1, 3, 1]] * 6,
         [[1, 3, 1]] * 3),

    "branches":

        ({"channel": 256, "kernel_size": 1, "stride": 1, "padding": 0},
         {"channel": 512, "kernel_size": 1, "stride": 2, "padding": 0},
         {"channel": 1024, "kernel_size": 1, "stride": 2, "padding": 0},
         {"channel": 2048, "kernel_size": 1, "stride": 2, "padding": 0})
}


class ResidualUnit(nn.HybridBlock):
    def __init__(self, l_chans, l_kernel_sizes, chan_params={None}, bbn=False, resize=False):
        """
        Generator for residual unit
        :param l_chans: list of channels for layers on trunk
        :param l_kernel_sizes: list of kernel sizes for layers on trunk
        :param l_strides: list of strides for layers on trunk
        :param chan_params: dict!! params for branch(shortcut) if needed.
        :param bbn: bool. switch for using shortcut in this unit.
        :param resize: bool. If this unit would resize input feat-map or not.
        """
        super(ResidualUnit, self).__init__()
        """

        """
        assert len(l_kernel_sizes) == len(l_chans)
        self.bbn = bbn
        self.branch = nn.HybridSequential()
        if bbn or self.bbn:
            self.branch.add(
                nn.Conv2D(channels=chan_params["channel"], kernel_size=chan_params["kernel_size"],
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

        self.trunk = nn.HybridSequential()
        for k in range(len(l_chans)):
            self.trunk.add(
                nn.Conv2D(channels=l_chans[k], kernel_size=l_kernel_sizes[k],
                          strides=1, padding=int(l_kernel_sizes[k] / 2)),
                nn.Activation(activation='relu'),
                nn.BatchNorm(in_channels=l_chans[k])
            )

        if resize:
            self.trunk[0]._kwargs["stride"] = (2, 2)
            self.branch[0]._kwargs["stride"] = (2, 2)

        self.final_relu = nn.Activation(activation='relu')

    def hybrid_forward(self, F, x, *args, **kwargs):
        res_shortcut = self.branch(x)
        res_trunk = self.trunk(x)
        res = self.final_relu(res_shortcut + res_trunk)
        # print(res.shape)
        return res


class ResidualLayer(nn.HybridBlock):
    def __init__(self, num_stages, ll_chans, ll_kernel_sizes, chan_param_onlybranch, **kwargs):
        """
        Generator for a residual unit
        :param num_stages: int,
            number of residual units in this residual layer
        :param ll_chans: list/tuple (of list),
            contains channel numbers for sublayers in each residual layer
        :param ll_kernel_sizes: list/tuple (of list),
            contains kernel sizes for sublayers in each residual layer
        :param chan_param_onlybranch: list/tuple,
            the param for the branch of this residual layer
        :param kwargs: ~
        """
        super(ResidualLayer, self).__init__(**kwargs)
        self.trunk = nn.HybridSequential()
        for k in range(num_stages):
            if k == 0:
                self.trunk.add(ResidualUnit(ll_chans[k], ll_kernel_sizes[k],
                                            chan_param_onlybranch, bbn=True, resize=True))
            else:
                self.trunk.add(ResidualUnit(ll_chans[k], ll_kernel_sizes[k]))

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.trunk(x)


class ResNet50(nn.HybridBlock):
    def __init__(self, params, IF_DENSE=True, **kwargs):
        """
        This is builder for a hybrid resnet-50 network
        :param params: parameters for resnet-50 network.
        :param IF_DENSE: set as true to add max-pooling and dense layer to the end of the network
        :param kwargs: not implemented yet
        """

        super(ResNet50, self).__init__(**kwargs)
        chans = params["channels"]
        kernel_sizes = params["kernel_sizes"]
        branches = params["branches"]
        assert len(chans) == len(kernel_sizes) & len(kernel_sizes) == len(branches)
        self.net = nn.HybridSequential()
        conv1 = nn.HybridSequential()
        conv1.add(
            nn.Conv2D(channels=64, kernel_size=7,
                      strides=2, padding=int(7 / 2)),
            nn.Activation(activation='relu'),
            nn.BatchNorm(in_channels=64)
        )
        self.net.add(conv1)
        for k in range(len(chans)):
            self.net.add(
                ResidualLayer(len(chans[k]), chans[k], kernel_sizes[k], branches[k])
            )

        if IF_DENSE:
            self.net.add(
                nn.MaxPool2D(),
                nn.Dense(1000)
            )

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.net(x)


def test():
    import time
    ctx = mx.gpu()
    x = nd.random.normal(0, 1, (1, 3, 224, 224), ctx=ctx)

    # Example for building a Residual Unit
    net = ResidualUnit([128, 128, 128], [3, 3, 3],
                       chan_params={"channel": 128, "kernel_size": 1, "stride": 1, "padding": 0},
                       bbn=False, resize=True)
    # print(net)

    # Example for building a Residual Layer conatining several units.
    len(params["channels"][0])
    net = ResidualLayer(3, params["channels"][0], params["kernel_sizes"][0],
                        params["branches"][0])
    # print(net)

    # Test for a Residual Network.
    net = ResNet50(params); xsym = mx.sym.Variable('data')
    net.hybridize()
    net.initialize(ctx=ctx)
    mx.viz.plot_network(net(xsym)).view() # view the network structure

    cnt = time.time()
    net(x)
    print(time.time() - cnt)

    print(net)
