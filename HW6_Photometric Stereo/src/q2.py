# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 22, 2021
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals, estimateShape 
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface 

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals
    
    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """

    B = None
    L = None
    # Your code here
    # Singular Value Decomposition of I
    u, sin, vh = np.linalg.svd(I, full_matrices = False)
    
    # Estimate the best rank-3 approximation
    # Set all other singular values to 0
    sin[3:] = 0
    s_sqrt = np.sqrt(np.diag(sin))

    # Estimate B and L : L.T @ B = I
    B = (s_sqrt @ vh)[0:3, :]
    L = (u @ s_sqrt).T[0:3, :]

    return B, L

def plotBasRelief(B, mu, nu, lam, suffix = ''):

    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter
    
    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """

    # Your code here
    # Form the G matrix
    G = np.array([[1, 0, 0], [0, 1, 0], [mu, nu, lam]])
    
    # Generalized bas-relief ambiguity
    new_B = np.linalg.inv(G).T @ B
    
    # Visulaize the reconstructed 3D depth map
    integrable_normals = enforceIntegrability(new_B, s)
    surface = estimateShape(integrable_normals, s)
    plotSurface(surface, suffix = suffix)


if __name__ == "__main__":

    # Part 2 (b)
    # Estimate B_hat and L_hat
    I, L0, s = loadData()
    B_hat, L_hat = estimatePseudonormalsUncalibrated(I)
    # Visualize the estimated albedos and normals
    albedos, normals = estimateAlbedosNormals(B_hat)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave('2b-a.png', albedoIm, cmap = 'gray')
    plt.imsave('2b-b.png', normalIm, cmap = 'rainbow')

    # Part 2 (c)
    # Compare L0 and L_hat
    print(f"Ground truth lighting L0 =\n{L0}")
    print(f"Estimated lighting L_hat =\n{L_hat}")

    # Part 2 (d)
    # Visualize the reconstructed 3D depth map
    surface = estimateShape(normals, s)
    plotSurface(surface, suffix = '2d')

    # Part 2 (e)
    # Transform non-integrable pseudonormals into integrable pseudonormals 
    integrable_normals = enforceIntegrability(B_hat, s)
    # Visualize the reconstructed 3D depth map
    surface = estimateShape(integrable_normals, s)
    plotSurface(surface, suffix = '2e')

    # Part 2 (f)
    # Visualize the corresponding surfaces
    vary_parameters = [-2, 1, 10]
    for i in vary_parameters:
        plotBasRelief(B_hat, i, 0, 1, suffix = f'_mu_{i}')
        plotBasRelief(B_hat, 0, i, 1, suffix = f'_nu_{i}')
        if i > 0:
            plotBasRelief(B_hat, 0, 0, i, suffix = f'_lam_{i}')