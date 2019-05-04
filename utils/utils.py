import numpy as np
import matplotlib.pyplot as plt

def aspResize(img, maxsize):
    """
    function for automatically resizing image while preserving original
    aspect ratio.
    :param img: input image
    :param maxsize: new maximum size for dst image. e.g. input 1920*1080,
        maxsize=1280 >> output 1280*720
    :return: scaled image
    """
