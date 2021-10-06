import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    sigma_est = skimage.restoration.estimate_sigma(image, multichannel=False)
    im2 = skimage.restoration.denoise_wavelet(image,3*sigma_est,multichannel=True)
    grey = skimage.color.rgb2gray(image)
    thresh = skimage.filters.threshold_otsu(grey)
    binary = (grey < thresh)#.astype(np.float)
    sem = skimage.morphology.square(7)
    open2 = skimage.morphology.closing(binary,sem)
    opened = skimage.segmentation.clear_border(open2)
    labels = skimage.measure.label(opened)

    for region in skimage.measure.regionprops(labels):
        if region.area >= 100:
            bboxes.append(region.bbox)
    bw = 1.0-binary
    bw = skimage.morphology.erosion(bw,skimage.morphology.square(3))
    return bboxes, bw