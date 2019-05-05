import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
from mxnet import autograd, init, contrib, nd, sym
from mxnet import gluon, image
from mxnet.gluon import utils as gutils
import os

def load_data_uav(data_dir = '../data/uav', batch_size=4, edge_size=256):
    # _download_pikachu(data_dir)
    train_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'train.rec'),
        path_imgidx = os.path.join(data_dir, 'train.idx'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),  # 输出图像的形状
        shuffle=True,  # 以随机顺序读取数据集
        rand_crop=1,  # 随机裁剪的概率为1
        min_object_covered=0.95, max_attempts=200)
    val_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'val.rec'), batch_size=batch_size,
        data_shape=(3, edge_size, edge_size), shuffle=False)
    return train_iter, val_iter

def im2data():
    # TODO: implement dataset from images
    pass
