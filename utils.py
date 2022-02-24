import numpy as np

def bounds(i, j, imshape):
    """
    :param i: index for dimension 0
    :param j: index for dimension 1
    :param imshape: (tuple) shape of the image
    :return: (bool) whether this point is in bounds of indexing the image
    """
    if (0 <= i < imshape[0]) and (0 <= j < imshape[1]):
        return True
    return False

def cumulativeCurveLength(contour):
    """
    Acquire the cumulative curve length of a contour
    :param contour: (ndarray), shape=(N, 2) ordered, each row is a coordinate of contour who's curveLength is needed
    :return: (ndarray), shape=(N,) dtype=float32 where cumulative[i] is the curvelength from contour[0] to contour[i] inclusive
    """
    cumulative = np.zeros(len(contour))

    for i in range(1, cumulative.shape[0]):
        base_sq = (contour[i][0] - contour[i - 1][0]) ** 2
        height_sq = (contour[i][1] - contour[i - 1][1]) ** 2
        dist = np.sqrt(base_sq + height_sq)
        cumulative[i] = cumulative[i - 1] + dist
    return cumulative
