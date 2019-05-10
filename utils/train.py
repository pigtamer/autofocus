import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from gluoncv import loss as gcvloss
from mxnet import autograd, init, contrib, nd, sym
from utils.utils import calc_loss, cls_eval, bbox_eval
from utils.coco_af import *
from main import predict
cls_lossfunc = gloss.SoftmaxCrossEntropyLoss()
# cls_lossfunc = gcvloss.FocalLoss()
bbox_lossfunc = gloss.L1Loss()




def training(data_iter, num_epoches, cls_lossfunc, bbox_lossfunc):
    # TODO: define the way that the model should be trained
    #   wth gluon.Trainer(...)
    for eph in range(num_epoches):
        pass
    pass


def validate(val_iter, net, ctx=mx.gpu()):
    idx, acc = 0 ,0
    val_iter.reset()
    for batch in val_iter:
        X = batch.data[0].as_in_context(ctx)
        Y = batch.label[0].as_in_context(ctx)
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
                iou_this = iou(bbox[0].asnumpy(), Y[0])
            if iou_this > 0:
                acc+=1
        idx += 1
    return acc/idx
