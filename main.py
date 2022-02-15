from maskToContour import MaskToContour
import numpy as np
from PIL import Image
import imageio

T = MaskToContour(debug=True)
mask = np.array(imageio.imread("/home/rylan/PycharmProjects/mask2contour/Re_Mask/predicted_001B-003080.20151209_img-02_image_lv_4ch.png"))
solid_mask = (mask == 170)*1
myo_mask = (mask == 85)*1
T(solid_mask, myo_mask)


