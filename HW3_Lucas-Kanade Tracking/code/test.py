import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from LucasKanade import LucasKanade
import scipy.ndimage
import cv2

# save = [29, 30, 59, 60, 89, 90, 119, 120]

# seq = np.load('../data/aerialseq.npy')

# for index in save:
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     plt.imshow(seq[:, :, index], cmap = 'gray')
#     plt.axis('off')

#     plots_dir = '../plots/'
#     if not os.path.exists(plots_dir):
#         os.makedirs(plots_dir)
#     fig.savefig('../plots/Aerial' + str(index) + '.png', bbox_inches = 'tight', pad_inches = 0)

# xx, yy = np.meshgrid(np.linspace(1, 3, num = 2, endpoint = True), np.linspace(2, 4, num = 2, endpoint = True))
# a, b = np.mgrid[1:3+1:2*1j,2:4+1:2*1j]
# print(xx, yy)
# print(a, b)

# a = np.arange(1, 5.1)
# print(a)

# seq = np.load("../data/carseq.npy")
# H, W, num_frames = seq.shape
# It1 = seq[:, :, 1]
# h1, w1 = It1.shape
# spline_temp = RectBivariateSpline(np.arange(h1), np.arange(w1), It1)
# spline_temp2 = RectBivariateSpline(np.linspace(0, h1, num = h1, endpoint = False), np.linspace(0, w1, num = w1, endpoint = False), It1)
# xx, yy = np.meshgrid(np.linspace(1, 2, num = 2, endpoint = True), np.linspace(1, 2, num = 2, endpoint = True))
# a, b = spline_temp.ev(yy, xx)
# print(a, b)
# c, d = spline_temp2.ev(yy, xx)
# print(c, d)

# x1, x2, y1, y2 = 1, 5, 2, 6
# w = x2-x1
# h = y2-y1
# xx, yy = np.meshgrid(np.linspace(x1, x2, num = w, endpoint = True), np.linspace(y1, y2, num = h, endpoint = True))
# xx2, yy2 = np.mgrid[x1:x2+1:w*1j,y1:y2+1:h*1j]
# print(xx, yy)
# print(xx2, yy2)

# rect = np.load('../results/girlseqrects.npy')
# print(rect.shape)
# seq = np.load("../data/girlseq.npy")
# print(seq.shape)

# w = 3
# h = 4
# xx, yy = np.meshgrid(np.arange(w), np.arange(h))
# # xx, yy = np.meshgrid(np.linspace(0, w, num = w, endpoint = False), np.linspace(0, h, num = h, endpoint = False))
# total_size = xx.shape[0] * xx.shape[1]
# xx1 = xx.reshape(1, total_size)
# yy1 = yy.reshape(1, total_size)
# xx2 = xx.flatten()
# yy2 = yy.flatten()
# X_homo = np.concatenate(([xx2], [yy2], np.ones((1, total_size))), axis = 0)

# print(xx1)
# print(xx2)
# print(X_homo)

# M = np.array([[1, 2, 3], [4, 5, 6]])
# rotate = M[:, 0:2]
# translate = M[:, 2]
# image1 = seq[:, :, 0]
# image1_warp01 = scipy.ndimage.affine_transform(input = image1, matrix = rotate, offset = translate, output_shape = None)        
# image1_warp02 = scipy.ndimage.affine_transform(input = image1, matrix = M, offset = 0.0, output_shape = None)
# print(image1_warp01)
# print(image1_warp02)
# print(image1_warp01 == image1_warp02)

# a = np.array([[1, 2, 3], [4, 5, 6]])
# print(a.reshape(1,-1))
# print(a.flatten())

# seq1 = np.load('../results/carseqrects.npy')
# print(seq1)
# seq2 = np.load('../results/carseqrects-wcrt.npy')
# print(seq2)
# seq3 = np.load('../results/girlseqrects.npy')
# print(seq3)
# seq4 = np.load('../results/girlseqrects-wcrt.npy')
# print(seq4)

seq = np.load('../data/carseq.npy')
img = seq[:, :, 0]
plt.imshow(img, cmap = 'gray')
plt.axis('off')
plt.show()
template = img
img_pyr_temp = [template]
for i in range(5):
    template = cv2.pyrDown(template)
    img_pyr_temp.append(template)
    plt.imshow(template, cmap = 'gray')
    plt.show()