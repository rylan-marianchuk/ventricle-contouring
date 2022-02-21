import numpy as np

def smooth(pointCloud):
    """

    :param pointCloud:
    :return:
    """
    for i in range(1, pointCloud.shape[0] - 1):
        dir = pointCloud[i+1] - pointCloud[i-1]
        new = pointCloud[i] + 0.5*dir
        pointCloud[i] = new
    return



def rejection(AB, Ah):
    a = Ah
    b = AB
    a2 = a - (np.dot(a, b) / np.dot(b, b)) * b
    return a2
