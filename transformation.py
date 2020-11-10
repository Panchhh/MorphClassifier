import cv2
import numpy as np
from functools import reduce
from skimage.feature import hog, local_binary_pattern
from sklearn.cluster import KMeans


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


class Sift:
    def __data_preparation(self):
        descriptor = []
        for data in self.dataset:
            kp, desc = self.sift.detectAndCompute(data, None)
            for d in desc:
                descriptor.append(d)
        return descriptor

    def __cluster_descriptors_model(self):
        descriptor = self.__data_preparation()
        model = KMeans(n_clusters=self.k, init='k-means++', max_iter=100, n_init=1, random_state=0)
        model.fit(descriptor)
        return model

    def __init__(self, dataset, k):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.dataset = dataset
        self.k = k
        self.model = self.__cluster_descriptors_model()

    def sift_extractor(self):
        def hist_feature(sift, trained_model, k, data):
            kp, des = sift.detectAndCompute(data, None)
            hist = np.zeros(k)
            nkp = np.size(kp)

            for d in des:
                idx = trained_model.predict([d])
                hist[idx] += 1 / nkp  # normalized
            return hist

        return lambda data: hist_feature(self.sift, self.model, self.k, data)
