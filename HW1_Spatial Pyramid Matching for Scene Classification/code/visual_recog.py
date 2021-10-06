import os
import math
import multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words
import matplotlib.pyplot as plt


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts: options
    * wordmap: numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    # Set up the parameters
    K = opts.K                                     # size of bins
    
    # Create the histogram from wordmap and Normalize 
    hist, label = np.histogram(wordmap, bins = np.arange(K+1))
    hist = hist / np.sum(hist)                     # Normalize

    # Plot the histogram to check
    # plt.bar(range(K), hist)
    # plt.title(f"K = {K}, histogram of aquarium/sun_aadolwejqiytvyne.jpg")
    # plt.show()

    return hist


def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts: options
    * wordmap: numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^(L+1)-1)/3)
    '''

    # Set up the parameters and Initialize hist_all
    K = opts.K
    L = opts.L
    row, col = wordmap.shape
    hist_all = np.array([], dtype = np.float32).reshape(1, 0)

    # Spatial Pyramid Matching : Slow Method (loop over each layers)
    # for layer_index in range(L):
    #     # Set up the weight of each layer
    #     if (layer_index == 0 or layer_index == 1):
    #         weight = 2 ** (-L)
    #     else:
    #         weight = 2 ** (L - layer_index - 1)

    #     # Chop the image into cells
    #     num_cell = 2 ** layer_index
    #     cell_row = int(row/num_cell)
    #     cell_col = int(col/num_cell)

    #     # Concatenate all histograms
    #     for row_index in range(num_cell):
    #         for col_index in range(num_cell):
    #             small_wordmap = wordmap[cell_row*row_index : cell_row*(row_index+1), cell_col*col_index : cell_col*(col_index+1)]
    #             single_hist = get_feature_from_wordmap(opts, small_wordmap)
    #             hist_all = np.append(hist_all, single_hist * weight)

    # Spatial Pyramid Matching : Fast Method (start from the finest layer and aggregate the others)
    # Start from the finest(top) layer
    # Set up the number of cells and size of each cell
    num_cell = 2 ** L
    cell_row = int(row/num_cell)
    cell_col = int(col/num_cell)

    # Initialize the finest layer and its weight
    finest_layer = np.zeros((num_cell, num_cell, K))
    if (L == 0 or L == 1):
        weight = 2 ** (-L)
    else:
        weight = 1/2

    # Compute the histograms of the finest layer 
    for row_index in range(num_cell):
        for col_index in range(num_cell):
            small_wordmap = wordmap[cell_row*row_index : cell_row*(row_index+1), cell_col*col_index : cell_col*(col_index+1)]
            single_hist = get_feature_from_wordmap(opts, small_wordmap)
            finest_layer[row_index, col_index, :] = single_hist
    hist_all = np.append(finest_layer.reshape(1,-1)[0] * weight, hist_all)

    # Aggregate the remaining layers
    for layer_index in range(L-1, -1, -1):
        # Set up the weight of each layer
        if (layer_index == 0 or layer_index == 1):
            weight = 2 ** (-L)
        else:
            weight = 2 ** (layer_index - L - 1)

        # Aggregate the remaining layers from the finest layer
        num_cell = 2 ** layer_index
        single_layer = np.zeros((num_cell, num_cell, K))
        for row_index in range(num_cell):
            for col_index in range(num_cell):
                single_layer[row_index, col_index, :] = np.sum(finest_layer[row_index*2 : (row_index+1)*2, 
                                                                col_index*2 : (col_index+1)*2, :], axis = (0, 1))
        hist_all = np.append(single_layer.reshape(1,-1)[0] * weight, hist_all)

    # Normalization    
    hist_all = hist_all / np.sum(hist_all)

    # Plot the histogram_all to check
    # plt.bar(range(hist_all.shape[0]), hist_all)
    # plt.title(f"K = {K}, L ={L}, size = {hist_all.shape[0]}, histogram_all of aquarium/sun_aadolwejqiytvyne.jpg")
    # plt.show()

    return hist_all
        

def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts: options
    * img_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^(L+1)-1)/3)
    '''

    # Load the image and check the data type and dimensions
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.tile(img, (1, 1, 3))

    # Extract the wordmap from the image (use dictionary)
    wordmap = visual_words.get_visual_words(opts, img, dictionary)

    # Compute the Spatial Pyramid Matching features (use wordmap)  
    feature = get_feature_from_wordmap_SPM(opts, wordmap)

    # Plot the feature to check
    # plt.bar(range(feature.shape[0]), feature)
    # plt.title(f"size = {feature.shape[0]}, SPM feature of aquarium/sun_aadolwejqiytvyne.jpg")
    # plt.show()
    
    return feature


def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from
    all training images.

    [input]
    * opts: options
    * n_worker: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    # Set up the file path and load the training files 
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    # Load the trainig files and labels 
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    training_img_num = len(train_files)
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # Multiprocessing to extract the traing features
    opts_list = [opts] * training_img_num
    img_path = [join(data_dir, img_name) for img_name in train_files]
    dictionary_list = [dictionary] * training_img_num
    args = zip(opts_list, img_path, dictionary_list)
    pool = multiprocessing.Pool(n_worker)
    features = pool.starmap(get_image_feature, args)

    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'), features = features, labels = train_labels,
    dictionary = dictionary, SPM_layer_num = SPM_layer_num)


def distance_to_set(word_hist, histograms):
    '''
    Compute distance between a histogram of visual words with all training
    image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K*(4^(L+1)-1)/3)
    * histograms: numpy.ndarray of shape (T,K*(4^(L+1)-1)/3)

    [output]
    * dis: numpy.ndarray of shape (T)
    '''
    
    # Compute the intersection similarity bectween word_hist and histograms  
    num_features, concantenated_size = histograms.shape
    intersection_similarity = np.minimum(word_hist, histograms)

    # Compute the distance (inverse of the intersection similarity) 
    dis = np.full((num_features), 1) - np.sum(intersection_similarity, axis = 1)

    return dis


def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the
    confusion matrix.

    [input]
    * opts: options
    * n_worker: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    # Set up file path and Load traind data
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    trained_features = trained_system['features']
    trained_labels = trained_system['labels']

    # Use the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    # Load the test data
    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_img_num = len(test_files)
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # Extract the features from test data
    opts_list = [opts] * test_img_num
    img_path = [join(data_dir, img_name) for img_name in test_files]
    dictionary_list = [dictionary] * test_img_num
    args = zip(opts_list, img_path, dictionary_list)
    pool = multiprocessing.Pool(n_worker)
    test_features = np.asarray(pool.starmap(get_image_feature, args))
    np.savez_compressed(join(out_dir, 'test_system.npz'), features = test_features)

    # Compute the predicted labels
    pred_labels = []
    for test_index in range(test_img_num):
        pred_index = np.argmin(distance_to_set(test_features[test_index, :], trained_features))
        pred_labels.append(trained_labels[pred_index])
    pred_labels = np.asarray(pred_labels)

    # Compute the Confusion Matrix and Accuracy
    confusion_matrix = np.zeros((8, 8))
    for true_index, pred_index in zip(test_labels, pred_labels):
        confusion_matrix[true_index][pred_index] += 1
    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

    return confusion_matrix, accuracy