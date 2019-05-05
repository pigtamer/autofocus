import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from mxnet import autograd, sym, init, nd, contrib
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import autograd, gluon, image, init, nd

def aspResize(img, maxsize):
    """
    function for automatically resizing image while preserving original
    aspect ratio.
    :param img: input image
    :param maxsize: new maximum size for dst image. e.g. input 1920*1080,
        maxsize=1280 >> output 1280*720
    :return: scaled image
    """


def calc_loss(cls_lossfunc, bbox_lossfunc,
              cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_lossfunc(cls_preds, cls_labels)
    bbox = bbox_lossfunc(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    # the result from class prediction is at the last dim
    # argmax() should be assigned with the last dim of cls_preds
    return (cls_preds.argmax(axis=-1) == cls_labels).sum().asscalar()


def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return ((bbox_labels - bbox_preds) * bbox_masks).abs().sum().asscalar()


def flatten_pred(pred):
    if len(pred.shape) != 4:
        pred.reshape(1, pred.shape[0], pred.shape[1], pred.shape[2])
    return pred.transpose((0, 2, 3, 1)).flatten()


def concat_preds(preds):
    return nd.concat(*[flatten_pred(p) for p in preds], dim=1)


def hybrid_flatten_pred(sym_pred):
    return sym.transpose(sym_pred, (0, 2, 3, 1)).flatten()


def hybrid_concat_preds(sym_preds):
    return sym.concat(*[hybrid_flatten_pred(p) for p in sym_preds], dim=1)


def fmap_grid(fmaplist):
    def factor(n):
        factors = set()
        for x in range(1, int(sqrt(n)) + 1):

            if n % x == 0:
                factors.add(x)
                factors.add(n // x)
        return sorted(factors)

    num_fmaps = len(fmaplist)
    w_grid = factor(num_fmaps)
    w_grid = int(w_grid[int((len(w_grid) - 1) / 2)])
    h_grid = int((num_fmaps / w_grid))

    fig, ax = plt.subplots(w_grid, h_grid, figsize=(w_grid, h_grid))
    fig.subplots_adjust(hspace=0, wspace=0)
    for i in range(w_grid):
        for j in range(h_grid):
            ax[i, j].xaxis.set_major_locator(plt.NullLocator())
            ax[i, j].yaxis.set_major_locator(plt.NullLocator())
            ax[i, j].imshow(fmaplist[int(w_grid * i + j)], cmap="bone")
    plt.show()

