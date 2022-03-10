from contour import MaskToContour
import numpy as np
import imageio
import os
import time

img_file = os.listdir("/home/rylan/testset_img_gt_mask/")[1]
mask_file = "predicted_" + img_file

GetContour = MaskToContour(debug=True)
mask = np.array(imageio.imread("/home/rylan/testset_img_gt_mask/" + mask_file))
img = np.array(imageio.imread("/home/rylan/testset_img_gt_mask/" + img_file))
solid_mask = (mask == 170)*1
myo_mask = (mask == 85)*1



start = time.time()
endo, epi, apex, quality = GetContour(solid_mask, myo_mask, img)
print("Time: " + str(time.time() - start))

