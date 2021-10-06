import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # Set up the initial M 
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # Compute the spline for interpolation
    h, w = It.shape
    h1, w1 = It1.shape
    spline_temp = RectBivariateSpline(np.arange(h), np.arange(w), It)
    spline_current = RectBivariateSpline(np.arange(h1), np.arange(w1), It1)

    # Compute the meshgrid of the template
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    # Compute A
    # Calculate the gradient: dI/dx and dI/dy
    dIdx = spline_temp.ev(yy, xx, dy = 1).flatten()
    dIdy = spline_temp.ev(yy, xx, dx = 1).flatten()
    # Calculate the jacobian: dW/dp
    dWdp1 = xx.flatten()
    dWdp2 = yy.flatten()
    dWdp3 = np.ones_like(xx.flatten())
    dWdp4 = xx.flatten()
    dWdp5 = yy.flatten()
    dWdp6 = np.ones_like(xx.flatten())
    # A = dI/dX * dW/dp
    A = np.zeros((xx.flatten().shape[0], 6))
    A[:, 0] = dIdx * dWdp1
    A[:, 1] = dIdx * dWdp2
    A[:, 2] = dIdx * dWdp3
    A[:, 3] = dIdy * dWdp4
    A[:, 4] = dIdy * dWdp5
    A[:, 5] = dIdy * dWdp6

    # Compute the Hessian = A.T @ A
    Hessian = A.T @ A

    # Iterations
    for iters in range(int(num_iters)):
        # Warp the image: compute X_bar
        xx_bar = M[0, 0] * xx + M[0, 1] * yy + M[0, 2] * np.ones_like(xx)
        yy_bar = M[1, 0] * xx + M[1, 1] * yy + M[1, 2] * np.ones_like(xx)

        # Check the valid area and set the error to 0 if invalid
        valid_index = (xx_bar >= 0) & (xx_bar[0] < w) & (yy_bar >= 0) & (yy_bar < h)
        valid_map = valid_index.astype(int)
        xx, yy = xx * valid_map, yy * valid_map
        xx_bar, yy_bar = xx_bar * valid_map, yy_bar * valid_map 
        
        # Compute I_temp and I_current
        I_temp = spline_temp.ev(yy, xx).flatten()
        I_current = spline_current.ev(yy_bar, xx_bar).flatten()
        
        # A and Hessian are precomputed
        # Compute b  
        b = (I_current - I_temp).reshape(-1, 1)

        # Solve the Least Square problem using Pseudo Inverse
        # Compute dp and dM
        dp = np.linalg.inv(Hessian) @ A.T @ b
        dM = np.array([[1 + dp[0, 0], dp[1, 0], dp[2, 0]],
                       [dp[3, 0], 1 + dp[4, 0], dp[5, 0]],
                       [0, 0, 1]])

        # Update M
        M = np.concatenate((M, [[0, 0, 1]]), axis = 0)
        M = M @ np.linalg.inv(dM)
        M = M[0:2, :]

        # Check dp
        if (np.linalg.norm(dp, ord = 2) < threshold):
            break 

    return M

