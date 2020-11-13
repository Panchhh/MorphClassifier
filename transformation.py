import cv2
import numpy as np
from functools import reduce
from skimage.feature import hog, local_binary_pattern
from sklearn.cluster import KMeans
from abc import ABC


def hog_extractor(orientations, pixels_per_cell, cells_per_block, multichannel):
    """
    Image to hog transformation function.

    :param orientations: HOG orientations
    :param pixels_per_cell: HOG pixels_per_cell
    :param cells_per_block: HOG cells_per_block
    :param multichannel: HOG multichannel

    :return: a function from image to HOG one-dimensional vector.
    """
    return lambda data: hog(data, orientations=orientations, pixels_per_cell=pixels_per_cell,
                            cells_per_block=cells_per_block, multichannel=multichannel)


def lbp_extractor(radius, n_points, method):
    """
    Image to lbp transformation function. Image is converted to grayscale before extracting lbp features.

    :param radius: LBP radius
    :param n_points: LBP n_points
    :param method: LBP method

    :return: a function from image to LBP image reshaped in a one-dimensional vector.
    """
    return compose(lambda data: np.reshape(local_binary_pattern(data, n_points, radius, method), -1),
                   cvt_color(cv2.COLOR_BGR2GRAY))


def lbp_hist_extractor(radius, n_points, method):
    """
    Image to LBP histogram transformation function. Image is converted to grayscale before extracting lbp features.
    The Histogram is quantized on the basis of the parameters radius and n_points.

    :param radius: LBP radius
    :param n_points: LBP n_points
    :param method: LBP method

    :return: a function from image to LBP histogram.
    """
    def lbp_hist(data):
        lbp = local_binary_pattern(data, n_points, radius, method)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        return hist
    return compose(lambda data: lbp_hist(data), cvt_color(cv2.COLOR_BGR2GRAY))


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


class BagOfWordFeature(ABC):

    def __init__(self, feature, dataset, k):
        self.feature = feature
        self.dataset = dataset
        self.k = k
        self.model = self.__cluster_descriptors_model()

    def data_preparation(self):
        descriptor = []
        for data in self.dataset:
            kp, desc = self.feature.detectAndCompute(data, None)
            for d in desc:
                descriptor.append(d)
        return descriptor

    def __cluster_descriptors_model(self):
        descriptor = self.data_preparation()
        model = KMeans(n_clusters=self.k, init='k-means++', max_iter=100, n_init=1, random_state=0)
        model.fit(descriptor)
        return model

    def feature_extractor(self):
        def hist_feature(feature, trained_model, k, data):
            kp, des = feature.detectAndCompute(data, None)
            hist = np.zeros(k)
            nkp = np.size(kp)

            for d in des:
                idx = trained_model.predict([d])
                hist[idx] += 1 / nkp  # normalized
            return hist

        return lambda data: hist_feature(self.feature, self.model, self.k, data)


class Sift(BagOfWordFeature):
    def __init__(self, dataset, k):
        super().__init__(cv2.xfeatures2d.SIFT_create(), dataset, k)


class Surf(BagOfWordFeature):
    def __init__(self, dataset, k):
        super().__init__(cv2.xfeatures2d_SURF.create(), dataset, k)
