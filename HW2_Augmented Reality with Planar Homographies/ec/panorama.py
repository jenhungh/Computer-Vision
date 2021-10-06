# Import necessary packages and functions
import sys
helper_function_path = '../python' 
sys.path.insert(1, helper_function_path)

import numpy as np
import cv2

from ar_helper import matchPics
from planarH import computeH_ransac

from opts import get_opts
opts = get_opts()

# Write script for Q4.1x
def panorama(img1, img2, opts):
    # Match the left and right images
    matches, locs1, locs2 = matchPics(img1, img2, opts)

    # Get the coordinates of the matching pairs
    # Extract the matching points
    locs1 = locs1[matches[:, 0]]
    locs2 = locs2[matches[:, 1]]
    # Flip x and y coordinates
    locs1[:, [0, 1]] = locs1[:, [1, 0]]
    locs2[:, [0, 1]] = locs2[:, [1, 0]]

    # Compute Homography Matrix using RANSAC
    H2to1, inliers = computeH_ransac(locs1, locs2, opts)

    # Warp image2 to image1
    # Know the size of the input images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # Compute the size of image2 after transfromation
    size_img2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype = np.float32).reshape(-1, 1, 2)
    trans_size_img2 = cv2.perspectiveTransform(size_img2, H2to1)
    # Set up the warping size (compute the boundary using trans_size_img2)
    warp_w = int(min(trans_size_img2[1][0][0], trans_size_img2[2][0][0]))
    warp_h = int(min(trans_size_img2[2][0][1], trans_size_img2[3][0][1])) - int(max(trans_size_img2[0][0][1], trans_size_img2[1][0][1]))
    # Warp image2
    warp_img2 = cv2.warpPerspective(img2, H2to1, dsize = (warp_w, warp_h))
    # Check warp image2
    # cv2.imshow('warp_img2', warp_img2)
    # cv2.waitKey(0)

    # Composite image1 with warp image2
    composite = warp_img2
    composite[:warp_h, :w1, :] = img1[:warp_h, :, :]

    return composite


# Main function
left = cv2.imread('left.jpg')
right = cv2.imread('right.jpg')
composite = panorama(left, right, opts)
cv2.imwrite('panorama.png', composite)