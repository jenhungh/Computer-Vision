from helper import briefMatch, computeBrief, detectCorners, convert2Gray, plotMatches
from planarH import computeH_ransac

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy


def matchPics(img1, img2, opts):
    # img1, img2 : Images to match
    # opts: input opts
    ratio = opts.ratio  # 'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  # 'threshold for corner detection using FAST feature detector'

    # Convert Images to GrayScale
    img1 = convert2Gray(img1)
    img2 = convert2Gray(img2)

    # Detect Features in Both Images
    locs1 = detectCorners(img1, sigma)
    locs2 = detectCorners(img2, sigma)

    # Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(img1, locs1)
    desc2, locs2 = computeBrief(img2, locs2)

    # Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio) 

    return matches, locs1, locs2


# Q2.1.5
def briefRotTest(img, opts):
    # in increments of 10 degrees from 0 to 360
    match_per_angle = []
    for angle in range(10, 361, 10):
        # Rotate Image
        rotate_img = scipy.ndimage.rotate(img, angle) 
        
        # Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(img, rotate_img, opts)

        # Display four orientations (10, 100, 200, and 350 degrees)
        if (angle == 10 or angle == 100 or angle == 200 or angle == 350):
            plotMatches(img, rotate_img, matches, locs1, locs2) 

        # Update histogram
        match_per_angle.append(len(matches))
        print(f"angle = {angle} degree, matching result = {len(matches)}")

    # Display histogram
    plt.bar([*range(10, 361, 10)], match_per_angle, width = 5, log = True)
    plt.title('Histogram of the count of matches for each orientation')
    plt.xlabel('orientations')
    plt.ylabel('counts of matches (log scaling)')
    plt.show()


def composeWarpedImg(img_source, img_target, img_replacement, opts):
    # Obtain the homography that warps img_source to img_target, then use it to overlay img_replacement over img_target
    # Obtain features from both source and target images
    matches, locs1, locs2 = matchPics(img_source, img_target, opts)

    # Extract the matching points
    locs1 = locs1[matches[:, 0]]
    locs2 = locs2[matches[:, 1]]
    # Flip x and y coordinates
    locs1[:, [0, 1]] = locs1[:, [1, 0]]
    locs2[:, [0, 1]] = locs2[:, [1, 0]]

    # Get homography by RANSAC
    H2to1, inliers = computeH_ransac(locs1, locs2, opts)
    H1to2 = np.linalg.inv(H2to1)

    # Create a composite image after warping the replacement image on top of the target image using the homography
    # Scale the replacement image properly
    img_replacement = cv2.resize(img_replacement, dsize = (img_source.shape[1], img_source.shape[0]))
    
    # Warp the replacement image
    hp_warp = cv2.warpPerspective(img_replacement, H1to2, dsize = (img_target.shape[1], img_target.shape[0]))
    # Check the replacement warpping
    # cv2.imshow('hp_warp', hp_warp)
    # cv2.waitKey(0)

    # Delete the target areas for composition
    delete_warp = cv2.warpPerspective(np.ones_like(img_replacement), H1to2, dsize = (img_target.shape[1], img_target.shape[0]))
    img_target_area_deleted = img_target - img_target * delete_warp
    # Check the image with target area deleted
    # cv2.imshow('img_target_area_deleted', img_target_area_deleted)
    # cv2.waitKey(0)

    # Composite the image
    composite_img = img_target_area_deleted + hp_warp 

    return composite_img