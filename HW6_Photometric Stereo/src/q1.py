# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 22, 2021
# ##################################################################### #

import numpy as np
from matplotlib import pyplot as plt
import skimage.io
from skimage.color import rgb2xyz
from utils import integrateFrankot, plotSurface

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a hemispherical bowl with a given center and radius. Assume that
    the hollow end of the bowl faces in the positive z direction, and the
    camera looks towards the hollow end in the negative z direction. The
    camera's sensor axes are aligned with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    [X, Y] = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    X = (X - res[0]/2) * pxSize * 1.e-4
    Y = (Y - res[1]/2) * pxSize * 1.e-4
    Z = np.sqrt(rad**2 + 0j - X**2 - Y**2)  
    Z = np.real(Z)

    image = None
    # Your code here
    # Build the normal vector
    n_vector = np.concatenate((X[:, :, np.newaxis], Y[:, :, np.newaxis], Z[:, :, np.newaxis]), axis = 2)
    N = n_vector.reshape(res[0]*res[1], -1)
    # Normalization
    norm = np.linalg.norm(N, ord = 2, axis = 1)
    N = N / norm.reshape(-1, 1)
    
    # Implement NdotL Algorithm
    L = light
    image = (N @ L).reshape(res[1], res[0])
    image[np.real(Z) == 0] = 0 

    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.npy.

    Parameters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    I = None
    L = None
    s = None
    
    # Your code here
    # Load the image and Compute I and L
    num_img = 7
    for i in range(1, num_img+1):
        # Load the image and check the datatype
        input_img_rgb = skimage.io.imread(path + f'input_{i}.tif')
        input_img_rgb = input_img_rgb.astype(np.uint16)

        # Convert the RGB images into the XYZ color space
        input_img_xyz = rgb2xyz(input_img_rgb)

        # Compute s
        h, w, _ = input_img_rgb.shape
        s = (h, w)
        
        # Compute L
        if I is None:
            I = np.zeros((7, h*w))
        I[i-1, :] = input_img_xyz[:, :, 1].reshape(1, h*w)

    # Compute I
    L = np.load(path + 'sources.npy')
    L = L.T       

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = None
    # Your code here
    # Least Square Problem : Ax = y 
    # L.T @ B = I
    A = L.T
    y = I
    # Pseudo-inverse : x = inv(A.T @ A) @ A.T @ y
    B = np.linalg.inv(A.T @ A) @ A.T @ y

    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = None
    normals = None
    # Your code here
    # albedos = the magnitudes of the pseudonormals
    albedos = np.linalg.norm(B, ord = 2, axis = 0)

    # normals = normalized normal vectors
    normals = B / albedos

    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f, g)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = None
    normalIm = None
    # Your code here
    # Reshape albedos
    albedoIm = albedos.reshape(s)
    
    # Rescale and Reshape normals
    normals = (normals + 1) / 2
    normalIm = normals.T.reshape(s[0], s[1], 3)

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (j)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    surface = None
    # Your code here
    # Rescale and Reshape normals
    # normals = (normals + 1) / 2
    normalIm = normals.T.reshape(s[0], s[1], 3)

    # Compute the partial derivatives
    n1, n2, n3 = normalIm[:, :, 0], normalIm[:, :, 1], normalIm[:, :, 2]
    df_dx = -n1 / n3
    df_dy = -n2 / n3
    
    # Estimate the actual surface 
    surface = integrateFrankot(df_dx, df_dy)

    return surface


if __name__ == '__main__':
    # Part 1(b)
    radius = 0.75 # cm
    center = np.asarray([0, 0, 0]) # cm
    pxSize = 7 # um
    res = (3840, 2160)

    light = np.asarray([1, 1, 1])/np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap = 'gray', vmin = 0, vmax = 1)
    plt.show()
    plt.imsave('1b-a.png', image, cmap = 'gray', vmin = 0, vmax = 1)

    light = np.asarray([1, -1, 1])/np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap = 'gray', vmin = 0, vmax = 1)
    plt.show()
    plt.imsave('1b-b.png', image, cmap = 'gray', vmin = 0, vmax= 1)

    light = np.asarray([-1, -1, 1])/np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap = 'gray', vmin = 0, vmax = 1)
    plt.show()
    plt.imsave('1b-c.png', image, cmap = 'gray', vmin = 0, vmax = 1)

    # Part 1(c)
    I, L, s = loadData('../data/')

    # Part 1(d) 
    # Singular Value Decomposition of I
    u, sin, vh = np.linalg.svd(I, full_matrices = False)
    print(f"Singular Value of I = {sin}")

    # Part 1(e)
    B = estimatePseudonormalsCalibrated(I, L)
    
    # # Part 1(f)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave('1f-a.png', albedoIm, cmap = 'gray')
    plt.imsave('1f-b.png', normalIm, cmap = 'rainbow')

    # # Part 1(i)
    surface = estimateShape(normals, s)
    plotSurface(surface, suffix = '1i')