import mxnet as mx
from mxnet.gluon import Block, HybridBlock, nn, loss as gloss, data as gdata
import cv2 as cv, matplotlib.pyplot as plt, sys, os

# TODO:
#   import mxnet ...
#   from symbols import mynet
#   from utils import det_fusion
#   
#   using custom deque or list
#
# input = imread(...)
#
# featnet = symbol.feat
# focusnet = symbol.focus
# chipnet (OR some operator) = symbol.chip
# detnet = symbols.det
#
# buffer = [None]
#
# iter_num = 3
#
# next = input
# det = [None] * iter_num
#
# for k in iter_num:
# 	buffer.pop(); buffer.append(next);
# 	feat = featnet(buffer[0]) # do not use this fucking buffer
#
# 	det_this = detnet(feat)
# 	det.append(det_this)
#
# 	focus = focusnet(feat)
# 	next = chipnet(focus)
#
# det_fusion(*det)