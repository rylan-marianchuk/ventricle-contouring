from maskToContour import MaskToContour
import numpy as np
from PIL import Image
import imageio
import os

img_file = os.listdir("./img/raw")[2]
mask_file = "predicted_" + img_file

GetContour = MaskToContour(debug=True)
mask = np.array(imageio.imread("./img/masks/" + mask_file))
img = np.array(imageio.imread("./img/raw/" + img_file))
solid_mask = (mask == 170)*1
myo_mask = (mask == 85)*1

endo, epi, apex = GetContour(solid_mask, myo_mask, img)


