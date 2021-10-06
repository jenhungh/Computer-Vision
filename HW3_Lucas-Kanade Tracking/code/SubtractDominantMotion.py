import numpy as np
from scipy.interpolate import RectBivariateSpline
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # Initialize the binary mask
    mask = np.zeros(image1.shape, dtype=bool)

    # Compute the spline for interpolation
    h1, w1 = image1.shape
    h2, w2 = image2.shape
    spline1 = RectBivariateSpline(np.arange(h1), np.arange(w1), image1)
    spline2 = RectBivariateSpline(np.arange(h2), np.arange(w2), image2)

    # Affine Warp
    # Compute the meshgrid of image1 
    xx, yy = np.meshgrid(np.arange(w1), np.arange(h1))
    # Compute transformation matrix M : LucasKanadeAffine or InverseCompositionAffine 
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    # M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    print(f"M = {M}")
    # Warp image1
    xx_bar = M[0, 0] * xx + M[0, 1] * yy + M[0, 2] * np.ones_like(xx) 
    yy_bar = M[1, 0] * xx + M[1, 1] * yy + M[1, 2] * np.ones_like(xx)
    
    # Check the valid area
    valid_index = (xx_bar >= 0) & (xx_bar < w1) & (yy_bar >= 0) & (yy_bar < h1)
    valid_map = valid_index.astype(int)
    xx, yy = xx * valid_map, yy * valid_map
    xx_bar, yy_bar = xx_bar * valid_map, yy_bar * valid_map 

    # Compute Image1_warp and Image2
    Image1_warp = spline1.ev(yy, xx)
    Image2 = spline2.ev(yy_bar, xx_bar)

    # Compute the subtraction
    diff = abs(Image2 - Image1_warp)
    mask[diff > tolerance] = 1

    return mask