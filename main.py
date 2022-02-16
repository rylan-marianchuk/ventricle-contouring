from maskToContour import MaskToContour
import numpy as np
from PIL import Image
import imageio

T = MaskToContour(debug=False)
mask = np.array(imageio.imread("/home/rylan/PycharmProjects/mask2contour/Re_Mask/predicted_003986.20160317Dcm_img-01_image_lv_4ch.png"))
solid_mask = (mask == 170)*1
myo_mask = (mask == 85)*1
T(solid_mask, myo_mask)


