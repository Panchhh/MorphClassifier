import cv2
import numpy as np
from functools import reduce
from skimage.feature import hog, local_binary_pattern


def hog_extractor(orientations, pixels_per_cell, cells_per_block, multichannel):
    """
    Image to hog transformation function.
    """
    return lambda data: hog(data, orientations=orientations, pixels_per_cell=pixels_per_cell,
                            cells_per_block=cells_per_block, multichannel=multichannel)


def lbp_extractor(radius, n_points, method):
    """
    Image to lbp transformation function. Image is converted to grayscale before extracting lbp features.
    """
    return compose(lambda data: np.reshape(local_binary_pattern(data, n_points, radius, method), -1),
                   cvt_color(cv2.COLOR_BGR2GRAY))


def cvt_color(code):
    return lambda data: cv2.cvtColor(data, code)


def resizer(shape):
    return lambda data: cv2.resize(data, shape)


def compose(*func):
    """
    It composes the functions applying them from the last to the first.

    :param func: Functions to be composed.
    :return: Composite function.
    """
    def comp(f, g):
        return lambda x: f(g(x))

    return reduce(comp, func, lambda x: x)
