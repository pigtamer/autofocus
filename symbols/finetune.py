import matplotlib.pyplot as plt
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd
from gluoncv import model_zoo, data, utils
from chips import focus_branch as foc


#
# with net.name_scope():
#     net.__delattr__("fc")
#     net.__delattr__("flat")
#     net.flat = nn.Activation('relu')
#     net.fc  = nn.BatchNorm()

def test(ctx=mx.cpu()):
    net = model_zoo.faster_rcnn_fpn_bn_resnet50_v1b_coco(pretrained=True, ctx = ctx)
    # net = model_zoo.resnet50_v1b(pretrained=True, ctx=ctx)
    im_fname = "/home/cunyuan/code/pycharm/data/uav/usc/1479/img/0131.jpg"
    x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)
    x = x.as_in_context(ctx)
    # x = nd.random.normal(0, 1, (30, 3, 512, 512), ctx=ctx)
    net.initialize(ctx=mx.cpu())
    net.hybridize()
    net(x)
    print(net)
    # box_ids, scores, bboxes = net(x)
    # ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)
    #
    # plt.show()


test(mx.cpu())
