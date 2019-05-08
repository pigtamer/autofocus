import matplotlib.pyplot as plt
import mxnet as mx
from symbols import resnet50, VGG
from mxnet.gluon import nn
from mxnet import nd, sym, autograd, image
from mxnet.gluon import loss as gloss
import argparse, time
from chips.focus_branch import *
from utils.coco_af import *
import os
import cv2 as cv
from gluoncv import model_zoo, data, utils

# parsing cli arguments
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load", dest="load",
                    help="bool: load model to directly infer rather than training",
                    type=int, default=1)
parser.add_argument("-b", "--base", dest="base",
                    help="bool: using additional base network",
                    type=int, default=0)
parser.add_argument("-e", "--epoches", dest="num_epoches",
                    help="int: trainig epoches",
                    type=int, default=50)
parser.add_argument("-bs", "--batch_size", dest="batch_size",
                    help="int: batch size for training",
                    type=int, default=8)
parser.add_argument("-is", "--imsize", dest="input_size",
                    help="int: input size",
                    type=int, default=256)

parser.add_argument("-lr", "--learning_rate", dest="learning_rate",
                    help="float: learning rate of optimization process",
                    type=float, default=0.01)
parser.add_argument("-opt", "--optimize", dest="optimize_method",
                    help="optimization method",
                    type=str, default="sgd")

parser.add_argument("-dp", "--data_path", dest="data_path",
                    help="str: the path to dataset",
                    type=str, default="../../data/uav/usc/1479/raw/")
# ../../data/uav/usc/1479/output/cropped/
parser.add_argument("-mp", "--model_path", dest="model_path",
                    help="str: the path to load and save model",
                    type=str, default="/home/cunyuan/code/pycharm/params/autofocus/")
parser.add_argument("-tp", "--test_path", dest="test_path",
                    help="str: the path to your test img",
                    type=str, default="../../data/uav/dji_island_4.mp4")
args = parser.parse_args()


def load_data_uav(data_dir='../data/uav', batch_size=4, edge_size=256):
    # _download_pikachu(data_dir)
    train_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'train.rec'),
        path_imgidx=os.path.join(data_dir, 'train.idx'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),  # 输出图像的形状
        shuffle=True,  # 以随机顺序读取数据集
        rand_crop=1,  # 随机裁剪的概率为1
        min_object_covered=0.95, max_attempts=200)
    val_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'val.rec'), batch_size=batch_size,
        data_shape=(3, edge_size, edge_size), shuffle=False)
    return train_iter, val_iter


# basenet = resnet50.ResNet50(params=resnet50.params, IF_DENSE=False)
# basenet = VGG.BaseNetwork(IF_TINY=False)
# net = nn.Sequential()
# net.add(
#     basenet,
#     FocusBranch()
# )
# net.initialize(ctx=mx.gpu())


# prenet = model_zoo.faster_rcnn_resnet50_v1b_coco(pretrained=True, ctx = mx.gpu())
# basenet = nn.HybridSequential()
# basenet.add(
#     prenet.features,
#     prenet.top_features
# )

prenet = model_zoo.resnet50_v1b(pretrained=True, ctx=mx.gpu())
basenet = nn.HybridSequential()
for layers in prenet._children:
    basenet.add(prenet._children[layers])
basenet = basenet[0:-3]

net = nn.HybridSequential()
net.add(
    basenet,
    HybridFocusBranch()
)
net[1].initialize(ctx=mx.gpu())
net[1].collect_params().setattr('lr_mult', 10)
net.hybridize()


batch_size, edge_size = args.batch_size, args.input_size
train_iter, val_iter = load_data_uav(args.data_path, batch_size, edge_size)
batch = train_iter.next()

trainer = mx.gluon.Trainer(net.collect_params(), args.optimize_method,
                           {'learning_rate': args.learning_rate, 'wd': 5E-4})

cls_loss = gloss.L1Loss()


def err_eval(bbox_preds, bbox_labels):
    return (bbox_labels - bbox_preds).abs().sum().asscalar()


if args.load:
    net.load_parameters(args.model_path + "Focuser-is256e50bs08-pResNet50-dUSC1479raw-lr0.01x10")
    # focustest()
else:
    lerr, lcnt = [], []
    for epoch in range(args.num_epoches):
        train_iter.reset()  # reset data iterator to read-in images from beginning
        start = time.time()
        err = 0;
        m = 0
        for batch in train_iter:
            X = batch.data[0].as_in_context(mx.gpu())
            Y = batch.label[0].as_in_context(mx.gpu())
            gts = lstlbl2bbox(Y, IF_COCO=True, orig_size=edge_size)
            with autograd.record():
                # generate anchors and generate bboxes
                fmap_genned = net(X)

                lbls = affine_fmap2gt(fmap_size=fmap_genned.shape,
                                      input_size=[edge_size] * 2,
                                      gts=gts,
                                      lthres=9 ** 2, rthres=64 ** 2)
                # plt.imshow(
                #     cv.resize(lbls[0, 0, :, :], (edge_size, edge_size)) * nd.sum(X[0, :, :, :], axis=0).asnumpy() / 3)
                # plt.show()
                l = cls_loss(fmap_genned, nd.array(lbls, ctx=mx.gpu()))
                if nd.sum(nd.array(lbls != 0)) == 0:
                    raise Exception("ERROR in labels.")
            l.backward()
            err += err_eval(fmap_genned, nd.array(lbls, ctx=mx.gpu()))
            m += fmap_genned.size
            trainer.step(batch_size)
        err /= m
        lcnt.append(time.time() - start); lerr.append(err)

        print("Round %d / %d, err %3.3f, cnt %3.3f" %
              (epoch + 1, args.num_epoches, err, time.time() - start))

        if (epoch + 1) % 5 == 0:
            net.save_parameters('Focuser')
    print(lcnt, "\n", lerr)

cap = cv.VideoCapture(args.test_path)
rd = 0
isize = 1024
while True:
    ret, frame = cap.read()
    img = nd.array(frame)
    feature = image.imresize(img, isize, isize).astype('float32')
    X = feature.transpose((2, 0, 1)).expand_dims(axis=0)

    countt = time.time()

    fmap = net(X.as_in_context(mx.gpu()))
    conn = calcConnect(fmap[0, 0, :, :].asnumpy(), gau_sigma=0,
                       thres_ratio=0.9, conn=1, IF_ABS=True)
    plt.imshow(cv.resize(conn[1].astype(np.float32), (isize, isize))
               * nd.sum(X[0, :, :, :], axis=0).asnumpy() / 3, cmap="gray")
    countt = time.time() - countt
    print("# %d     SPF: %3.2f" % (rd, countt))
    cv.imshow("focus", cv.resize(cv.resize(conn[1].astype(np.float32), (isize, isize))
               * nd.sum(X[0, :, :, :], axis=0).asnumpy() / 3 / 255, (1280,720)))
    cv.waitKey(10);
    rd += 1
