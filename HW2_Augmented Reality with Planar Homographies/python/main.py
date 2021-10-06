import cv2
from helper import plotMatches
from opts import get_opts

from ar_helper import matchPics, briefRotTest, composeWarpedImg
from planarH import computeH, computeH_norm, computeH_ransac 


def main():
    opts = get_opts()

    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')

    # Q2.1.4
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
    # Display matched features
    plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

    # Q2.1.6
    # briefRotTest(cv_cover, opts)

    # Check Q2.2.1, Q2.2.2, and Q2.2.3
    # Extract the matching points
    # locs1 = locs1[matches[:, 0]]
    # locs2 = locs2[matches[:, 1]]
    # Flip x and y coordinates
    # locs1[:, [0, 1]] = locs1[:, [1, 0]]
    # locs2[:, [0, 1]] = locs2[:, [1, 0]]
    
    # Q2.2.1
    # H2to1 = computeH(locs1, locs2)
    # print(f"H2to1 = {H2to1}")
    # Q2.2.2
    # H2to1 = computeH_norm(locs1, locs2)
    # print(f"H2to1 = {H2to1}")
    # Q2.2.3
    # best_H2to1, inliers = computeH_ransac(locs1, locs2, opts)
    # print(f"best_H2to1 = {best_H2to1}")
    # print(f"inliers = {inliers}")

    # Q2.2.4
    # composite_img = composeWarpedImg(cv_cover, cv_desk, hp_cover, opts)
    # cv2.imwrite('../result/hp_desk.png', composite_img)


if __name__ == '__main__':
    main()