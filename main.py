import matplotlib.pyplot as plt
import mxnet as mx
from symbols import resnet50, VGG
from mxnet.gluon import nn
from mxnet import nd, sym, autograd, image
from mxnet.gluon import loss as gloss
import argparse, time
from chips.focus_branch import *
from utils.coco_af import *
# from utils.train import *
import os
from det import ssd_module as detmod, anchor_params as ach_params
import cv2 as cv
from utils.utils import cls_eval, bbox_eval, calc_loss
from gluoncv import model_zoo

ctx = mx.gpu()
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
                    type=int, default=10)
parser.add_argument("-bs", "--batch_size", dest="batch_size",
                    help="int: batch size for training",
                    type=int, default=32)
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
                    type=str, default="../data/uav/usc/1479/cropped/")
# ../../data/uav/usc/1479/output/cropped/
parser.add_argument("-mp", "--model_path", dest="model_path",
                    help="str: the path to load and save model",
                    type=str, default="../params/autofocus/")
parser.add_argument("-tp", "--test_path", dest="test_path",
                    help="str: the path to your test img",
                    type=str, default="../data/video233.avi")
args = parser.parse_args()


def load_data_uav(data_dir='../data/uav', batch_size=4, edge_size=256):
    # _download_pikachu(data_dir)
    train_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'train.rec'),
        path_imgidx=os.path.join(data_dir, 'train.idx'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),  # shape of output image
        shuffle=True,  # read data in randomly
        rand_crop=1,  # prob of cropping randomly
        min_object_covered=0.95, max_attempts=200)
    val_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'val.rec'), batch_size=1,
        data_shape=(3, 1024, 1024), shuffle=False)
    return train_iter, val_iter


prenet = model_zoo.resnet50_v1b(pretrained=True, ctx=mx.gpu())
basenet = nn.HybridSequential()
for layers in prenet._children:
    basenet.add(prenet._children[layers])
basenet = basenet[0:-3]

net0 = nn.HybridSequential()
net0.add(
    basenet,
    HybridFocusBranch()
)

# net0.load_parameters("../params/autofocus/Focuser-is256e50bs08-pResNet50-dUSC1479raw-lr0.01x10", ctx=mx.gpu())
net0.load_parameters("./chips/Focuser", ctx=mx.gpu())

basenet = net0[0]
focusnet = net0[1]

det_branch = detmod.LightSSD(num_cls=1, num_ach=ach_params.num_anchors, IF_TINY=True)
det_branch.load_parameters("/home/deeplearning/cunyuan/pycharm/myssd/myssd.params",
                           ctx=mx.gpu())


class detnet(nn.HybridBlock):
    def __init__(self, base, bfoc, bdet, lr_mult={"focus": 10, "detection": 20}, **kwargs):
        super().__init__()
        self.basenet = base
        self.focus_branch = bfoc
        self.detect_branch = bdet

    def hybrid_forward(self, F, x, *args, **kwargs):
        # focus_area = self.basenet(x)
        # for k in rds:
        #     det_this_scl = self.detect_branch(focus_area)
        #     focus_area = self.focus_branch(self.basenet(x))
        # x = x.as_in_context(mx.gpu())
        fmap = self.focus_branch(self.basenet(x))
        conn = calcConnect(fmap[0, 0, :, :].asnumpy(), gau_sigma=1,
                           thres_ratio=0.9, conn=1, IF_ABS=True)
        focus_area, area_coord, focus_loc = genChip(x[0, :, :, :].asnumpy(), conn, (128, 128))
        det_this_scl = self.detect_branch(nd.array(
            cv.resize(focus_area[0].transpose(2, 1, 0), (256, 256)).
                transpose(2, 1, 0)).expand_dims(0))
        return (focus_area, focus_loc, area_coord, det_this_scl)


net = detnet(base=basenet,
             bfoc=focusnet,
             bdet=det_branch)

net._children["detect_branch"].initialize(ctx=mx.gpu())
net._children["detect_branch"].hybridize()
net._children["focus_branch"].collect_params().setattr('lr_mult', 2)
net._children["focus_branch"].hybridize()
net._children["detect_branch"].collect_params().setattr('lr_mult', 10)

batch_size, edge_size = args.batch_size, args.input_size
train_iter, val_iter = load_data_uav(args.data_path, batch_size, edge_size)
batch = train_iter.next()

trainer = mx.gluon.Trainer(net.collect_params(), args.optimize_method,
                           {'learning_rate': args.learning_rate, 'wd': 5E-4})


def predict(X, wf=256, hf=256, Wf=1024, Hf=1024):
    focus_chips, focus_locs, area_coord, det_res = net(X)
    chip_xmin, chip_ymin = area_coord[0][0] / Wf, area_coord[0][1] / Hf
    anchors, cls_preds, bbox_preds = det_res
    cls_probs = cls_preds.softmax().transpose((0, 2, 1))
    output = nd.contrib.MultiBoxDetection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0].asscalar() != -1]
    if idx == []: return nd.array([[0, 0, 0, 0, 0, 0, 0]])
    output = output[0, idx]
    output[:, 2] *= hf / Hf
    output[:, 4] *= hf / Hf
    output[:, 3] *= wf / Wf
    output[:, 5] *= wf / Wf

    output[:, 2] += chip_xmin
    output[:, 4] += chip_xmin
    output[:, 3] += chip_ymin
    output[:, 5] += chip_ymin
    return output, area_coord


def display(img, output, chip, frame_idx=0, threshold=0, show_all=0):
    lscore = []
    for row in output:
        lscore.append(row[1].asscalar())
    for row in output:
        score = row[1].asscalar()
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
        if score == max(lscore):
            cv.rectangle(img, (bbox[0][0].asscalar(), bbox[0][1].asscalar()),
                         (bbox[0][2].asscalar(), bbox[0][3].asscalar()),
                         (1. * (1 - score), 1. * score, 1. * (1 - score)),
                         int(10 * score))
            print("%s\t%s\t%s\t%s\t%s"%(bbox[0][0].asscalar(), bbox[0][1].asscalar(),
                                        bbox[0][2].asscalar(), bbox[0][3].asscalar(), frame_idx))
            cv.putText(img, "f%s:%3.2f" % (frame_idx, score), org=(bbox[0][0].asscalar(), bbox[0][1].asscalar()),
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0))
        if show_all:
            cv.rectangle(img, (bbox[0][0].asscalar(), bbox[0][1].asscalar()),
                         (bbox[0][2].asscalar(), bbox[0][3].asscalar()),
                         (1. * (1 - score), 1. * score, 1. * (1 - score)),
                         int(10 * score))
        # cv.rectangle(img, (chip[0][0], chip[0][1]), (chip[0][2], chip[0][3]), (1, 1, 1), 2)
        cv.imshow("res", img)
    cv.waitKey(10)


cls_loss = gloss.SoftmaxCrossEntropyLoss()
bbox_loss = gloss.L1Loss()
def validate(val_iter, net, ctx=mx.gpu()):
    idx, acc = 0 ,0
    val_iter.reset()
    for batch in val_iter:
        X = batch.data[0].as_in_context(ctx).astype('float32')
        Y = batch.label[0].as_in_context(ctx).astype('float32')
        # generate anchors and generate bboxes
        output, chip = predict(X, 128, 128)
        lscore = []
        for row in output:
            lscore.append(row[1].asscalar())
        for row in output:
            score = row[1].asscalar()
            h, w = X.shape[-1],X.shape[-2]
            bbox = [row[2:6] * nd.array((w, h, w, h), ctx=row.context)]
            if score == max(lscore):
                iou_this = iou(bbox[0].asnumpy(), Y[0][0][1:].asnumpy()*1024)
        if iou_this > 0:
            acc+=1
        idx += 1
    return acc/idx
if args.load:
    net._children["detect_branch"].load_parameters("../params/ssd/myssd5.params")
    ap = validate(val_iter, net)
    print(ap)

    # vid = 2666;
    # im = "0001";
    # dp = "../data/uav/usc/";
    # isize = 1024
    # fname = dp + "%s/" % vid + "img/%s.jpg" % im
    # frame = cv.imread(fname)

    cap = cv.VideoCapture(args.test_path)
    rd = 0;
    isize = 1024
    while True:
        rd += 1
        ret, frame = cap.read()
        # if rd < 500: continue
        img = nd.array(frame)
        img = image.imresize(img, isize, isize)
        feature = img.astype('float32')
        # print(feature.shape)
        X = feature.transpose((2, 0, 1)).expand_dims(axis=0)
        X = X.as_in_context(mx.gpu())
        output, chip = predict(X, 128, 128)
        display(cv.resize(frame, (1280, 720)), output, chip, frame_idx=rd, threshold=0)

    # net.load_parameters(args.model_path + "Focuser-is256e50bs08-pResNet50-dUSC1479raw-lr0.01x10")
    # focusplot(net, 3200, 1029, thr=0.9, dp="../../data/uav/usc/")
    print("pause here")
else:
    for epoch in range(args.num_epoches):
        acc_sum, mae_sum, n, m = 0.0, 0.0, 0, 0
        train_iter.reset()  # reset data iterator to read-in images from beginning
        start = time.time()
        for batch in train_iter:
            X = batch.data[0].as_in_context(ctx)
            Y = batch.label[0].as_in_context(ctx)
            with autograd.record():
                # generate anchors and generate bboxes
                focus_chips, focus_locs, area_coords, det_res = net(X)
                anchors, cls_preds, bbox_preds = det_res
                # print(net)

                # assign classes and bboxes for each anchor
                bbox_labels, bbox_masks, cls_labels = nd.contrib.MultiBoxTarget(anchor=anchors, label=Y,
                                                                                cls_pred=cls_preds.transpose((0, 2, 1)))
                # calc loss
                l = calc_loss(cls_loss, bbox_loss, cls_preds, cls_labels,
                              bbox_preds, bbox_labels, bbox_masks)
            l.backward()
            trainer.step(batch_size)
            acc_sum += cls_eval(cls_preds, cls_labels)
            n += cls_labels.size
            mae_sum += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
            m += bbox_labels.size
        print('epoch %2d, class err %.2e, bbox mae %.2e, time %.1f sec' % (
            epoch + 1, 1 - acc_sum / n, mae_sum / m, time.time() - start))
        # Checkpoint
        if (epoch + 1) % 5 == 0:
            net.export('FPN')

