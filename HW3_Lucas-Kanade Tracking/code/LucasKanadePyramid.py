import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def LucasKanadePyramid(It, It1, rect, threshold, num_iters, num_levels, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param num_levels: number of pyramid levels to use
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    # Create the Image Pyramid
    template = It
    img_pyr_temp = [template]
    for i in range(num_levels):
        template = cv2.pyrDown(template)
        img_pyr_temp.append(template) 

    current = It1
    img_pyr_current = [current]
    for i in range(num_levels):
        current = cv2.pyrDown(current)
        img_pyr_current.append(current)

    # Iterations
    for iters in range(int(num_iters)):
        for level in reversed(range(num_levels)):
            # Set up the initial parameter guess and the corners of the rectangle 
            p = p0
            # Print out rectangle to check
            x1, y1, x2, y2 = rect
            print(f"rect = {x1}, {y1}, {x2}, {y2}")

            # Compute the spline for sub-pixel interpolation
            h, w = img_pyr_temp[level].shape
            h1, w1 = img_pyr_current[level].shape
            spline_temp = RectBivariateSpline(np.arange(h), np.arange(w), img_pyr_temp[level])
            spline_current = RectBivariateSpline(np.arange(h1), np.arange(w1), img_pyr_current[level])

            # Compute I_temp
            xx, yy = np.meshgrid(np.arange(x1, x2 + 0.1), np.arange(y1, y2 + 0.1))
            total_size = xx.shape[0] * xx.shape[1]
            I_temp = spline_temp.ev(yy, xx)

            # Compute I_current
            I_current = spline_current.ev(yy + p[1], xx + p[0])

            # Calculate the gradient: dI/dx and dI/dy 
            dIdx = spline_current.ev(yy + p[1], xx + p[0], dy = 1).flatten()
            dIdy = spline_current.ev(yy + p[1], xx + p[0], dx = 1).flatten()
            
            # Calculate the jacobian: dW/dp
            dWdp = np.eye(2)
            
            # Compute A and b
            # Compute A = dI/dX * dW/dp
            A = np.zeros((total_size, 2))
            for index in range(total_size):
                A[index, 0] = dIdx[index] * dWdp[0, 0] + dIdy[index] * dWdp[1, 0]
                A[index, 1] = dIdx[index] * dWdp[0, 1] + dIdy[index] * dWdp[1, 1]
            # Compute b  
            b = (I_temp - I_current).reshape(-1, 1)

            # Solve the Least Square problem using Pseudo Inverse
            dp = np.linalg.inv(A.T @ A) @ A.T @ b

            # Update p
            p = np.asarray([p[0] + dp[0, 0], p[1] + dp[1, 0]])

            # Update p0
            p0 = p 

        # Check the norm of dp
        if (np.linalg.norm(dp, ord = 2) < threshold):
            break

    return p