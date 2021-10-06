"""
Homework4.
Replace 'pass' by your implementation.
"""

import numpy as np
import scipy.ndimage
import util
import scipy 

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Scale the data
    pts1 = pts1 / M
    pts2 = pts2 / M
    
    # Form matrix A
    x1, y1 = pts1[:, 0].reshape(-1, 1), pts1[:, 1].reshape(-1, 1)
    x2, y2 = pts2[:, 0].reshape(-1, 1), pts2[:, 1].reshape(-1, 1)
    one = np.ones((x1.shape[0], 1))
    A = np.concatenate((x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, one), axis = 1)

    # Singular Value Decomposition
    u, s, vh = np.linalg.svd(A)
    F = vh.T[:, -1].reshape(3, 3)

    # Refine and Singularize F
    F = util.refineF(F, pts1, pts2)

    # Unscale the Fundamental Matrix
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    F = T.T @ F @ T

    return F       

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Form the Essential Matrix
    E = K2.T @ F @ K1
    return E

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Initialize P and error
    N = pts1.shape[0]
    P = np.zeros((N, 3))
    err = 0
    
    # Extract 2D values
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]
    
    # Triangulation
    for i in range(N):
        # Form matrix A
        A = np.asarray([x1[i] * C1[2, :] - C1[0, :],
                        y1[i] * C1[2, :] - C1[1, :],
                        x2[i] * C2[2, :] - C2[0, :],
                        y2[i] * C2[2, :] - C2[1, :]])

        # Singular Value Decomposition
        u, s, vh = np.linalg.svd(A)
        p = vh.T[:, -1]
        # Normalization
        p = p / p[-1]
        # Update P
        P[i, :] = p[0:3]

        # Reprojection
        p1 = C1 @ p 
        p2 = C2 @ p
        # Normailization
        p1 = p1 / p1[-1]
        p2 = p2 / p2[-1]
        # Compute and Update reprojection error
        error = np.sum((p1[0:2] - pts1[i, :])**2) + np.sum((p2[0:2] - pts2[i, :])**2) 
        err = err + error

    return P, err

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Apply Gaussian Filter to both images 
    img1_filter = scipy.ndimage.gaussian_filter(im1, sigma = 1)
    img2_filter = scipy.ndimage.gaussian_filter(im2, sigma = 1)

    # Find the epipolar line on image2
    p1 = np.array([x1, y1, 1]).reshape(-1, 1)
    epi_line = F @ p1
    # Get the coefficients : ax + by + c = 0 
    a, b, c = epi_line

    # Find the possible matches along the epi_line 
    search = 40
    poss_y = np.arange(y1 - search, y1 + search)
    poss_x = (- c - b * poss_y) / a
    poss_x = poss_x.astype(int)
    # Check the validity (depend on window size)
    H, W, D = im2.shape
    window = 10
    half_w = window//2 
    valid = (poss_x >= half_w) & (poss_x < W-half_w) & (poss_y >= half_w) & (poss_y < H-half_w) 
    poss_x, poss_y = poss_x[valid], poss_y[valid]

    # Correspondence Matching
    error = np.inf
    for i in range(poss_x.shape[0]):
        # Possible corresponding points on image2
        p2_x, p2_y = poss_x[i], poss_y[i]
        
        # Compute the window similarity
        window1 = img1_filter[y1-half_w:y1+half_w+1, x1-half_w:x1+half_w+1, :]
        window2 = img2_filter[p2_y-half_w:p2_y+half_w+1, p2_x-half_w:p2_x+half_w+1, :]
        dis = np.sum((window1 - window2) ** 2)

        # Find the closest correspondences
        if dis < error:
            error = dis
            x2, y2 = p2_x, p2_y
    
    return x2, y2

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    # Initialize max inliers
    N = pts1.shape[0]
    max_inliers = 0
    
    # RANSAC Algorithm
    for iter in range(nIters):
        # Print out iteration index
        print(f"iteration = {iter+1}")

        # Initialize inliers 
        current_inliers = np.zeros(N, dtype = np.bool)

        # Randomly select 8 points to compute F 
        # np.random.seed()
        # sample = np.random.choice(N, size = 8, replace = False)
        sample = np.random.choice(N, size = 8)
        pts1_sample = pts1[sample, :]
        pts2_sample = pts2[sample, :]

        # Compute the Fundamental Matrix using Eight Point Algorithm 
        current_F = eightpoint(pts1_sample, pts2_sample, M)

        # Compute the epipolar line
        p1_homo = np.concatenate((pts1, np.ones((N, 1))), axis = 1)
        p2_pred = (current_F @ p1_homo.T).T

        # Compute the euclidean distance between epipolar line and pts2 
        p2_homo = np.concatenate((pts2, np.ones((N, 1))), axis = 1)
        factor = np.sqrt(np.sum(p2_pred[:, 0:2] ** 2, axis = 1))
        dis = abs(np.sum(p2_pred * p2_homo, axis = 1)) / factor

        # Update current inliers
        current_inliers[sample] = True        
        current_inliers[dis < tol] = True
        # Compute the number of inliers
        num_inliers = np.sum(current_inliers) 

        # Update F and inliers if needed
        if (num_inliers > max_inliers):
            max_inliers = num_inliers
            F = current_F
            inliers = current_inliers
        
        # Print max_inliers to check
        print(f"max_inliers = {max_inliers}")
  
    return F, inliers

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Compute the rotation angle theta
    theta = np.sqrt(np.sum(r ** 2))
    
    # Deal with corner case : no rotation
    if theta == 0:
        k = r
    else:
        k = r / theta
    
    # Compute the cross-product matrix K
    k1, k2, k3 = k[:, 0]
    K = np.array([[0, -k3, k2],
                  [k3, 0, -k1],
                  [-k2, k1, 0]])

    # Apply Rodrigues Rotation Formula
    # R = I + sin(theta) * K + (1-cos(theta)) * K^2  
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    return R

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    '''
    Reference: https://www2.cs.duke.edu/courses/compsci527/fall13/notes/rodrigues.pdf
    '''
    # Define A, rho, s, and c
    A = (R - R.T) /2
    rho = np.array([A[2, 1], A[0, 2], A[1, 0]]).reshape(-1, 1)
    s = np.sqrt(np.sum(rho ** 2))
    c = (np.sum(np.diag(R)) - 1) / 2

    # s = 0 and c = 1
    if s == 0 and c == 1:
        r = np.zeros((3, 1))
    
    # s = 0 and c = -1
    elif s ==0 and c == -1:
        # Compute v
        R_plus_I = R + np.eye(3)
        for col in range(3):
            if np.sum(R_plus_I[:, col]) != 0:
                v = R_plus_I[:, col]
                break
        # Compute u and r
        u = v / np.sqrt(np.sum(v ** 2))
        r = u * np.pi

        # Distinguish r or -r
        r1, r2, r3 = r[:, 0] 
        if np.sqrt(np.sum(r ** 2)) == np.pi and ((r1 == 0 and r2 == 0 and r3 < 0) or (r1 == 0 and r2 < 0) or (r1 < 0)):
            r = -r
        else:
            r = r 

    # remaining cases
    else:
        u = rho / s
        theta = np.arctan2(s, c)
        r = u * theta

    return r

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Extract w, r, and t
    N = p1.shape[0]
    w = x[0: -6].reshape(N, 3)
    w_homo = np.concatenate((w, np.ones((N, 1))), axis = 1)
    r2 = x[-6: -3].reshape(3, 1)
    t2 = x[-3:].reshape(3, 1)

    # Compute C1 and C2
    C1 = K1 @ M1
    M2 = np.concatenate((rodrigues(r2), t2), axis = 1)
    C2 = K2 @ M2

    # Compute estimated projections
    p1_est = C1 @ w_homo.T
    p2_est = C2 @ w_homo.T
    p1_hat = p1_est.T / p1_est.T[:, -1].reshape(-1, 1)
    p2_hat = p2_est.T / p2_est.T[:, -1].reshape(-1, 1)

    # Compute Residuals
    residuals = np.concatenate([(p1-p1_hat[:, :2]).reshape([-1]), (p2-p2_hat[:, :2]).reshape([-1])])

    return residuals
 
'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            w, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Set up the x for rodriguesResidual
    r2_init = invRodrigues(M2_init[:, 0:3]).reshape(-1, 1)
    t2_init = M2_init[:, 3].reshape(-1, 1)
    w_init = P_init.reshape(-1, 1)
    x_init = np.concatenate((w_init, r2_init, t2_init), axis = 0).reshape([-1])

    # Define the residual function
    def residual_func(x):
        return rodriguesResidual(K1, M1, p1, K2, p2, x)

    # Apply Least Square Optimizer to solve for x
    x = scipy.optimize.leastsq(residual_func, x_init)
    x = x[0]

    # Extract w, r2, and t2
    N = p1.shape[0]
    w = x[0: -6].reshape(N, 3)
    r2 = x[-6: -3].reshape(3, 1)
    t2 = x[-3:].reshape(3, 1)

    # Build M2 using r2 and t2
    R2 = rodrigues(r2)
    M2 = np.concatenate((R2, t2), axis = 1)

    return M2, w    