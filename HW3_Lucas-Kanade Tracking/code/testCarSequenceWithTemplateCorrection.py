import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries
# Set up iterations and thresholds
parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

# Load the video frames and Set up the reported frames 
seq = np.load("../data/carseq.npy")
reported_frames = [0, 99, 199, 299, 399]

# Set up initial guess, initial template, and first frame
# The first frame is used to correct the drift 
p0 = np.zeros(2)
template = seq[:, :, 0]
template_first = seq[:, :, 0]

# Set up the initial rectangle and the rectangle history   
rect_initial = [59, 116, 145, 151]
rect = rect_initial
rect_history = []
rect_history.append(rect)

# Apply Lucas-Kanade tracker with Template Correction to every frames
num_frames = seq.shape[2]
for index in range(num_frames-1):
    # Print out the frame index
    print(f"frame = {index+1}")

    # Load the current frames  
    It1 = seq[:, :, index+1]

    # Apply Lucas-Kanade tracker with Template Correction
    # Compute p
    p = LucasKanade(template, It1, rect, threshold, num_iters, p0)
    # Compute pn
    pn_x = (rect[0] + p[0]) - rect_initial[0]
    pn_y = (rect[1] + p[1]) - rect_initial[1]
    pn = np.array([pn_x, pn_y])
    # Compute pn_star
    pn_star = LucasKanade(template_first, It1, rect_initial, threshold, num_iters, pn)

    # Template Correction : Update the rectangle
    # Naive Update
    if (np.linalg.norm(pn_star - pn, ord = 2) <= template_threshold):
        # Update the rectangle : drift correction
        p_current_x = pn_star[0] - (rect[0] - rect_initial[0])
        p_current_y = pn_star[1] - (rect[1] - rect_initial[1])
        rect = [rect[0] + p_current_x, rect[1] + p_current_y, rect[2] + p_current_x, rect[3] + p_current_y]
        # Update template
        template = seq[:, :, index+1]
        # Reset p0
        p0 = np.zeros(2)
    # No Update
    else:
        p0 = p
    
    # Append the rectangle
    rect_history.append(rect)

    # Plot the reported frames
    rect_without_correction = np.load('../results/carseqrects.npy')
    if index in reported_frames:
        # Plot frames
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(seq[:, :, index], cmap = 'gray')
        plt.axis('off')
        # Plot the rectangles without correction
        x1_wo, y1_wo, x2_wo, y2_wo = rect_without_correction[index, :]
        width_wo, height_wo = (x2_wo - x1_wo), (y2_wo - y1_wo) 
        rectangle_wo = patches.Rectangle((x1_wo, y1_wo), width_wo, height_wo, edgecolor = 'b', facecolor = 'None', linewidth = 3)
        ax.add_patch(rectangle_wo)
        # Plot the rectangles with correction
        x1, y1, x2, y2 = rect_history[index] 
        rectangle = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1), edgecolor = 'r', facecolor = 'None', linewidth = 2)
        ax.add_patch(rectangle)

        # Save the figures
        plots_dir = '../plots/'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        fig.savefig('../plots/Car-wcrt_f' + str(index+1) + '.png', bbox_inches = 'tight', pad_inches = 0)

# Save the rectangle history
results_dir = '../results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
np.save('../results/carseqrects-wcrt.npy', np.asarray(rect_history))