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

def cumulative_curve_length(contour):
    """
    Acquire the cumulative curve length of a contour
    :param contour: (ndarray), shape=(N, 2) ordered, each row is a coordinate of contour whose curveLength is needed
    :return: (ndarray), shape=(N,) dtype=float32 where cumulative[i] is the curvelength from contour[0] to contour[i] inclusive
    """
    return np.hstack((np.zeros(1), np.cumsum(np.linalg.norm(np.diff(contour, axis=0), axis=1))))
    #cumulative = np.zeros(len(contour))

    #for i in range(1, cumulative.shape[0]):
        #base_sq = (contour[i][0] - contour[i - 1][0]) ** 2
        #height_sq = (contour[i][1] - contour[i - 1][1]) ** 2
        #dist = np.sqrt(base_sq + height_sq)
        #cumulative[i] = cumulative[i - 1] + dist
    #return cumulative

def contour_density_distance(contour):
    """
    Assume contour points are equi-distant, but what is that distance? Useful to know for quality purposes
    :param contour: (ndarray), shape=(N, 2) ordered, each row is a coordinate of contour
    :return: (float) the distance between each point in the contour
    """
    return np.linalg.norm(contour[0] - contour[1])


def normal_from_point(i, contour, centroid):
    """
    Return the normal to point contour[i], pointing away from centroid
    :param i: the index of the point to return normal from
    :param contour: (ndarray), shape=(N, 2) ordered, each row is a coordinate of contour
    ::param centroid: (ndarray) shape=(2,) the coordinate of the ventricle's centroid
    :return: (ndarray) shape=(2,) the normal pointing exterior to this contour
    """
    # If this is a left edge point, its left normal is the zero vector
    if i - 1 < 0:
        normal_l = np.array([0, 0])
    else:
        normal_l = np.flip(contour[i - 1] - contour[i]) * [-1, 1]

        # Flip the normal if its pointing the wrong way. The dot product with the vector from i to the centroid
        # should be negative, otherwise negate it
        if np.dot(normal_l, centroid - contour[i]) > 0:
            normal_l *= -1

    # Same as above for right size
    if i + 1 >= contour.shape[0]:
        normal_r = np.array([0, 0])
    else:
        normal_r = np.flip(contour[i + 1] - contour[i]) * [-1, 1]
        if np.dot(normal_r, centroid - contour[i]) > 0:
            normal_r *= -1
    return ((normal_r + normal_l) / np.linalg.norm(normal_r + normal_l)) * 1.002


def first_flood_fill_size(myo_mask):
    """
    Get the size of the first non-zero continuous shape in the myo mask
    :param myo_mask: (ndarray), shape=(N, M), dtype=int  1 assigned to only pixels on the lining (myocardium) of the ventricle, 0 elsewhere
    :return: (int) size of the first encountered non-zero continuous shape in the myo mask
    """
    myo_copy = myo_mask.copy()
    start = np.argwhere(myo_copy == 1)[0]
    Q = [start]
    covered = set()
    while len(Q) > 0:
        i,j = Q.pop()
        myo_copy[i,j] = 0
        covered.add((i,j))
        for mi, mj in zip([-1, -1, -1, 0, 0, 1, 1, 1], [-1, 0, 1, -1, 1, -1, 0, 1]):
            if bounds(i + mi, j + mj, myo_copy.shape) and myo_copy[i+mi, j+mj] == 1 and (i + mi, j + mj) not in covered:
                Q.append((i + mi, j + mj))

    return len(covered)


