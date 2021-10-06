import os
import multiprocessing
from os.path import join, isfile

import numpy as np
import scipy.ndimage
import skimage.color
from skimage import io
from PIL import Image
from sklearn.cluster import KMeans

from opts import get_opts
opts = get_opts()

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts: options
    * img: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    # Check data type and range
    if (type(img[0, 0, 0]) != np.float32):
        img = img.astype(np.float32) / 255
    if (np.amax(img) > 1.0 or np.amin(img) < 0.0):
        img = img.astype(np.float32) / 255

    # Get the size of the image 
    img_size = img.shape
    row, col, channel = img_size[0], img_size[1], img_size[2]

    # Make sure there are 3 channels (Duplicate gray-scale images)
    if len(img_size) == 2:
        img = img[:, :, np.newaxis]
        img = np.tile(img, (1, 1, 3))
    elif channel > 3:
        img = img[:, : , :3]

    # Convert image into lab color space
    lab_img = skimage.color.rgb2lab(img)

    # Set up filter scales and filter responses
    filter_scales = opts.filter_scales
    filter_responses = np.zeros((row, col, 3*4*len(filter_scales)))
    
    # Update filter responses
    for s_index in range(len(filter_scales)):
        for c_index in range(3):
            filter_responses[:, :, 3*4*s_index + c_index] = scipy.ndimage.gaussian_filter(lab_img[:, :, c_index], filter_scales[s_index])
            filter_responses[:, :, 3*4*s_index + 3 + c_index] = scipy.ndimage.gaussian_laplace(lab_img[:, :, c_index], filter_scales[s_index])
            filter_responses[:, :, 3*4*s_index + 6 + c_index] = scipy.ndimage.gaussian_filter(lab_img[:, :, c_index], filter_scales[s_index], order = [0, 1])
            filter_responses[:, :, 3*4*s_index + 9 + c_index] = scipy.ndimage.gaussian_filter(lab_img[:, :, c_index], filter_scales[s_index], order = [1, 0])

    return filter_responses


def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to
    disk. This is a worker function called by compute_dictionary.

    Your are free to make your own interface based on how you implement
    compute_dictionary.
    '''

    # Set up the input information of args and read the image
    img_index, alpha, img_path = args
    img = io.imread(img_path)
    img = img.astype(np.float32) / 255

    # Extract the filter responses
    filter_responses = extract_filter_responses(opts, img)
    row, col, F = filter_responses.shape
    T = row * col
    
    # Randomly sampled the filter responses
    sampled_index = np.random.randint(T, size = alpha)
    sampled_filter_responses = filter_responses.reshape(T, F)
    sampled_filter_responses = sampled_filter_responses[sampled_index, :]

    # Save to a temporary file
    feat_dir = opts.feat_dir 
    os.makedirs(feat_dir, exist_ok = True)
    np.save(join(feat_dir, f"img{img_index}.npy"), sampled_filter_responses)


def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts: options
    * n_worker: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''

    # Set up file path and parameters
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha = opts.alpha
    filter_scales = opts.filter_scales
    # Set up the training files path
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()  
    img_path = [join(data_dir, img_name) for img_name in train_files]

    # Multiprocess the training data
    img_index = range(1, len(img_path)+1)
    alpha_list = [alpha] * len(img_path)
    args = zip(img_index, alpha_list, img_path)
    pool = multiprocessing.Pool(n_worker)
    pool.map(compute_dictionary_one_image, args)

    # Collect all subprocess to form the filter responses
    filter_responses = np.array([], dtype = np.float32).reshape(0, 3*4*len(filter_scales))
    for img in os.listdir(feat_dir):
        subprocess_responses = np.load(join(feat_dir, img))
        filter_responses = np.append(filter_responses, subprocess_responses, axis = 0)

    # Apply K-means to cluster the responses 
    kmeans = KMeans(n_clusters = K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_ 

    # Save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)


def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of
    visual words.

    [input]
    * opts: options
    * img: numpy.ndarray of shape (H,W) or (H,W,3)
    * dictionary: numpy.ndarray of shape (K,3F)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''

    # Initialize the size of wordmap
    img_size = img.shape
    row, col = img_size[0], img_size[1]
    wordmap = np.zeros((row, col))

    # Compute every pixel of the wordmap : Slow Method (loop over row and col for every pixel)
    # filter_responses = extract_filter_responses(opts, img)
    # for i in range(row):
    #     for j in range(col):
    #         pixel = np.array(filter_responses[i, j, :]).reshape(1,-1)
    #         distance = scipy.spatial.distance.cdist(pixel, dictionary, metric = 'euclidean')
    #         wordmap[i, j] = np.argmin(distance)

    # Compute every pixel of the wordmap : Fast Method (reshape filter responses)
    filter_responses = extract_filter_responses(opts, img)
    filter_responses = filter_responses.reshape((row * col), -1)
    distance = scipy.spatial.distance.cdist(filter_responses, dictionary, metric = 'euclidean')
    wordmap = np.argmin(distance, axis = 1).reshape(row, col)

    return wordmap