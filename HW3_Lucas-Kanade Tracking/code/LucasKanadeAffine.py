import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # Set up the initial affine matrix M
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # Compute the spline for interpolation
    h, w = It.shape
    h1, w1 = It1.shape
    spline_temp = RectBivariateSpline(np.arange(h), np.arange(w), It)
    spline_current = RectBivariateSpline(np.arange(h1), np.arange(w1), It1)

    # Compute the meshgrid of the template
    XX, YY = np.meshgrid(np.arange(w), np.arange(h))

    # Iterations
    for iters in range(int(num_iters)):
        # Warp the image: compute X_bar
        xx, yy = XX, YY
        xx_bar = M[0, 0] * xx + M[0, 1] * yy + M[0, 2] * np.ones_like(xx)
        yy_bar = M[1, 0] * xx + M[1, 1] * yy + M[1, 2] * np.ones_like(xx)

        # Check the valid area
        valid_index = (xx_bar >= 0) & (xx_bar[0] < w) & (yy_bar >= 0) & (yy_bar < h)
        xx, yy = xx[valid_index], yy[valid_index]
        xx_bar, yy_bar = xx_bar[valid_index], yy_bar[valid_index]

        # Compute I_temp and I_current
        I_temp = spline_temp.ev(yy, xx).flatten()
        I_current = spline_current.ev(yy_bar, xx_bar).flatten()

        # Calculate the gradient: dI/dx and dI/dy 
        dIdx = spline_current.ev(yy_bar, xx_bar, dy = 1).flatten()
        dIdy = spline_current.ev(yy_bar, xx_bar, dx = 1).flatten()
        
        # Calculate the jacobian: dW/dp
        dWdp1 = xx.flatten()
        dWdp2 = yy.flatten()
        dWdp3 = np.ones_like(xx.flatten())
        dWdp4 = xx.flatten()
        dWdp5 = yy.flatten()
        dWdp6 = np.ones_like(xx.flatten())

        # Compute A and b
        # Compute A = dI/dX * dW/dp
        A = np.zeros((xx.flatten().shape[0], 6))
        A[:, 0] = dIdx * dWdp1
        A[:, 1] = dIdx * dWdp2
        A[:, 2] = dIdx * dWdp3
        A[:, 3] = dIdy * dWdp4
        A[:, 4] = dIdy * dWdp5
        A[:, 5] = dIdy * dWdp6
        # Compute b  
        b = (I_temp - I_current).reshape(-1, 1)

        # Solve the Least Square problem using Pseudo Inverse
        dp = np.linalg.inv(A.T @ A) @ A.T @ b

        # Update M
        M[0, :] += dp[0:3, 0]
        M[1, :] += dp[3:6, 0]

        # Check the norm of dp
        if (np.linalg.norm(dp, ord = 2) < threshold):
            break 

    return M
