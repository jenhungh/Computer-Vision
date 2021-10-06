import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    # Split the rows based on min_row and max_row
    # Sort the bboxes based on max_row
    bboxes.sort(key = lambda x : x[2])
    
    # Initialize rows and single_row
    all_rows, single_row = [], []
    
    # Split the characters into rows 
    boundary = bboxes[0][2]
    for bbox in bboxes:
        # Extract min_row and max_row
        min_row, max_row = bbox[0], bbox[2]
        # Check for new rows
        if min_row > boundary:
            # Sort the previos row based on min_column
            single_row.sort(key = lambda x : x[1])
            # Update rows
            all_rows.append(single_row)
            # Update boundary and Reset single_row 
            boundary = max_row
            single_row = []
        # Update single_row
        single_row.append(bbox)
    # Sort and Update the final single_row into rows
    single_row.sort(key = lambda x : x[1])
    all_rows.append(single_row) 

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    # Crop the bounding box
    crop_bboxes, row_crop_bboxes = [], []
    for r in range(len(all_rows)):
        for i, bbox in enumerate(all_rows[r]):
            # Crop the bbox
            min_row, min_col, max_row, max_col = bbox
            crop_bbox = bw[min_row:max_row, min_col:max_col]
            # Padding
            crop_bbox = np.pad(crop_bbox, ((30, 30), (30, 30)), 'constant', constant_values = (1, 1))
            # Resize and Preprocessing (erosion)
            crop_bbox = skimage.transform.resize(crop_bbox, (32, 32))
            crop_bbox = skimage.morphology.erosion(crop_bbox, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
            # skimage.io.imsave('../crop_images/%s_row%s_letter%s.png'%(img.split('.')[0], r+1, i+1), crop_bbox)
            # Transpose, Flatten, and Update
            crop_bbox = crop_bbox.T
            crop_bbox = crop_bbox.reshape(1, -1)
            row_crop_bboxes.append(crop_bbox)
        # Update crop_bboxes and Reset row_crop_bboxes
        crop_bboxes.append(row_crop_bboxes)
        row_crop_bboxes = []
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    ##########################
    # Print the image name
    print("\n" + img.split('.')[0] + "\n")

    # Classify the letter in crop_bboxes
    for r in range(len(crop_bboxes)):
        for x in crop_bboxes[r]:
            # Forward propagation
            h1 = forward(x, params, 'layer1')
            probs = forward(h1, params, 'output', softmax)
            # Evaluate the one-hot vector of the letter
            letter_index = np.argmax(probs, axis = 1)
            # Transfer index into letters
            letter = letters[letter_index][0]
            # Print out the letters
            print(f"{letter} ", end = '')
        print("\n")  