import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage.feature
from helper import briefMatch, computeBrief, detectCorners, convert2Gray, plotMatches
from planarH import computeH_ransac

import scipy

PATCHWIDTH = 9

# def convert2Gray(img):
#     # Convert image to grayscale
#     if len(img.shape) == 3:
#         img_gray = skimage.color.rgb2gray(img)
#     else:
#         img_gray = img.astype(np.float64) / 255.0

#     return img_gray

# def detectCorners(img, sigma):
#     # fast method
#     result_img = skimage.feature.corner_fast(
#         img, n=PATCHWIDTH, threshold=sigma)
#     locs = skimage.feature.corner_peaks(result_img, min_distance=1)
#     return locs

# def makeTestPattern(patchWidth, nbits):
#     np.random.seed(0)
#     compareX = patchWidth*patchWidth * np.random.random((nbits, 1))
#     compareX = np.floor(compareX).astype(int)
#     np.random.seed(1)
#     compareY = patchWidth*patchWidth * np.random.random((nbits, 1))
#     compareY = np.floor(compareY).astype(int)

#     return (compareX, compareY)


# def computeBrief(locs):
#     patchWidth = 9
#     nbits = 256
#     compareX, compareY = makeTestPattern(patchWidth, nbits)
#     m, n = img.shape
#     # print(m, n)

#     halfWidth = patchWidth//2
#     # print(halfWidth)

#     locs = np.array(list(filter(lambda x: halfWidth <=
#                                 x[0] < m-halfWidth and halfWidth <= x[1] < n-halfWidth, locs)))

#     return locs

# img = cv_cover = cv2.imread('../data/cv_cover.jpg')
# img = convert2Gray(img)
# locs = detectCorners(img, 0.15)
# print(locs.shape)
# locs = computeBrief(locs)
# print(locs.shape)
# print(44//9)

# a = np.array([[0, 0], [1, 0]])
# b = np.array([[1, 2], [3, 4]])
# print(a[:, 0])
# print(b[a[:, 1]])


# a = np.array([[1, 1], [2, 2], [4, 4]])
# b = np.array([[1], [2], [4]])
# print(a/b)

a = np.array([1, 2, 3, 4, 5, 6])
b = np.random.choice(a, 3)
print(b)