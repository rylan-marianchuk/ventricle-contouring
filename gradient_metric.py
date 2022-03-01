import numpy as np
import plotly.express as px

def sum_underlying_gradient(img, endo_contour, epi_contour):
    """

    :param img: (ndarray) shape=(N, M, 3) the vanilla MR image
    :param endo_contour: (ndarray) shape=(density, 2), ordered, each row a coordinate on the image as a part of the contour
    :param epi_contour: (ndarray) shape=(density, 2), ordered, each row a coordinate on the image as a part of the contour
    :return: (float) sum of the gradient values underlying endo_contour, (float) sum of the gradient values underlying epi_contour
    """
    gradient = np.gradient(img)[0]
    endo_sum = 0
    epi_sum = 0
    for i in range(endo_contour.shape[0]):
        endo_sum += np.abs(gradient[round(endo_contour[i,1]), round(endo_contour[i,0])].sum())
        epi_sum += np.abs(gradient[round(epi_contour[i,1]), round(epi_contour[i,0])].sum())
    return endo_sum, epi_sum
