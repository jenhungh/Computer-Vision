import numpy as np

def computeH(locs1, locs2):
    # Q2.2.1
    # Compute the homography between two sets of points
    # Form Matrix A to solve for h
    # Initialize the size of A
    num_match = locs1.shape[0]
    A = np.zeros((2*num_match, 9))
    
    # A corresponding pair provides two constraints
    for index in range(num_match):
        # Extract corresponding points (x1, y1) and (x2, y2) 
        x1, y1 = locs1[index, :]
        x2, y2 = locs2[index, :]
        # Form A based on Q1.2(3)
        A[2*index, :] = [-x2, -y2, -1, 0, 0, 0, x1*x2, x1*y2, x1]
        A[2*index+1, :] = [0, 0, 0, -x2, -y2, -1, y1*x2, y1*y2, y1]
    
    # Apply SVD to solve for h
    u, s, vh = np.linalg.svd(A)
    h = vh.T[:, -1]

    # Change h into H 
    H2to1 = h.reshape(3,3)

    return H2to1


def computeH_norm(locs1, locs2):
    # Q2.2.2
    # Compute the centroid of the points
    locs1_mean = np.mean(locs1, axis = 0)
    locs2_mean = np.mean(locs2, axis = 0)

    # Shift the origin of the points to the centroid
    locs1 = locs1 - locs1_mean
    locs2 = locs2 - locs2_mean

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    max_dis1 = np.max(np.sqrt(np.sum(locs1*locs1, axis = 1)))
    locs1 = (np.sqrt(2) / max_dis1) * locs1
    max_dis2 = np.max(np.sqrt(np.sum(locs2*locs2, axis = 1)))
    locs2 = (np.sqrt(2) / max_dis2) * locs2

    # Similarity transform 1
    T1 = np.array([[(np.sqrt(2) / max_dis1), 0, -(np.sqrt(2) / max_dis1) * locs1_mean[0]], 
                   [0, (np.sqrt(2) / max_dis1), -(np.sqrt(2) / max_dis1) * locs1_mean[1]], 
                   [0, 0, 1]])

    # Similarity transform 2
    T2 = np.array([[(np.sqrt(2) / max_dis2), 0, -(np.sqrt(2) / max_dis2) * locs2_mean[0]], 
                   [0, (np.sqrt(2) / max_dis2), -(np.sqrt(2) / max_dis2) * locs2_mean[1]], 
                   [0, 0, 1]])

    # Compute homography using normalized locs
    H2to1_norm = computeH(locs1, locs2)

    # Denormalization
    H2to1 = np.dot(np.dot(np.linalg.inv(T1), H2to1_norm), T2)

    return H2to1


def computeH_ransac(locs1, locs2, opts):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points
    # the number of iterations to run RANSAC for
    max_iters = opts.max_iters
    # the tolerance value for considering a point to be an inlier
    inlier_tol = opts.inlier_tol

    # Get the number of matching points
    num_match = locs1.shape[0]
    print(f"num of matches = {num_match}")

    # Change locs1 and ocs2 into Homogeneous Coordinates : for error checking
    locs1_homo = np.hstack((locs1, np.ones((num_match, 1))))
    locs2_homo = np.hstack((locs2, np.ones((num_match, 1))))

    # Initialize max inliers
    max_inliers = 0

    # RANSAC Algorithm
    for iter in range(max_iters):
        # Randomly select 4 point pairs to compute H 
        np.random.seed()
        sample = np.random.choice(num_match, size = 4, replace = False)
        locs1_sample = locs1[sample]
        locs2_sample = locs2[sample]
        H = computeH_norm(locs1_sample, locs2_sample)

        # Compute the error of predicted locs1
        # Compute predicted locs1 (Dimension: 3*N)
        locs1_pred = np.dot(H, locs2_homo.T)
        # Normalize predicted locs1 (Dimension: N*3)
        locs1_pred = locs1_pred.T / locs1_pred.T[:, -1].reshape(num_match, 1)
        # Compute the error and the euclidean distance
        error = locs1_homo - locs1_pred
        dis = np.sqrt(np.sum(error*error, axis = 1))

        # Initialize and Update current inliers
        current_inliers = np.zeros((num_match, 1))
        current_inliers[sample] = 1        
        current_inliers[dis < inlier_tol] = 1
        # Compute the number of inliers
        num_inliers = np.sum(current_inliers) 

        # Update best H and inliers if needed
        if (num_inliers > max_inliers):
            max_inliers = num_inliers
            best_H2to1 = H
            inliers = current_inliers   

    return best_H2to1, inliers