import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries
# Set up iterations, threshold, and tolerance 
parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

# Load the video frames and Set up the reported frames
seq = np.load('../data/aerialseq.npy')
reported_frames = [29, 59, 89, 119]

# Apply Subtract Dominant Motion to every frames
frames = seq.shape[2]
for index in range(frames-1):
    # Print out the frame index
    print(f"frame = {index+1}")
    
    # Load images
    image1 = seq[:, :, index] 
    image2 = seq[:, :, index+1]

    # Apply Subtract Dominant Motion
    mask = SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance)

    # Extarct the moving objects
    target = np.where(mask == 1)

    # Plot the reported frames
    if index in reported_frames:
        # Plot frames
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(image1, cmap = 'gray')
        plt.axis('off')
        
        # Plot the moving objects
        plt.plot(target[1], target[0], 'bo', markersize = 2)

        # Save the reported figures
        plots_dir = '../plots/'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        fig.savefig('../plots/Aerial_f' + str(index+1) + '.png', bbox_inches = 'tight', pad_inches = 0)