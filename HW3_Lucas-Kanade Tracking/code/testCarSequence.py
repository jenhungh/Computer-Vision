import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from LucasKanade import LucasKanade 

# write your script here, we recommend the above libraries
# Set up iterations and threshold 
parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

# Load the video frames and Set up the reported frames 
seq = np.load("../data/carseq.npy")
reported_frames = [0, 99, 199, 299, 399]

# Set up the initial rectangle and the rectangle history   
rect = [59, 116, 145, 151]
rect_history = []
rect_history.append(rect)

# Apply Lucas-Kanade tracker to every frames
num_frames = seq.shape[2]
for index in range(num_frames-1):
    # Print out the frame index
    print(f"frame = {index+1}")
    
    # Load template and current frames 
    It = seq[:, :, index] 
    It1 = seq[:, :, index+1]

    # Apply Lucas-Kanade tracker
    p = LucasKanade(It, It1, rect, threshold, num_iters)

    # Update and Append rectangle
    rect = [rect[0] + p[0], rect[1] + p[1], rect[2] + p[0], rect[3] + p[1]]
    rect_history.append(rect)

    # Plot the reported frames
    if index in reported_frames:
        # Plot frames
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(It, cmap = 'gray')
        plt.axis('off')
        
        # Plot the rectangles
        x1, y1, x2, y2 = rect_history[index]
        width, height = (x2 - x1), (y2 - y1)
        rectangle = patches.Rectangle((x1, y1), width, height, edgecolor = 'r', facecolor = 'none', linewidth = 3)
        ax.add_patch(rectangle)

        # Save the figures
        plots_dir = '../plots/'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        fig.savefig('../plots/Car_f' + str(index + 1) + '.png', bbox_inches = 'tight', pad_inches = 0)

# Save the rectangle history
results_dir = '../results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
np.save('../results/carseqrects.npy', np.asarray(rect_history))