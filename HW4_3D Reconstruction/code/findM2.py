'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

# Import necessary package
import numpy as np
import matplotlib.pyplot as plt
import helper
import submission
import os

# Load the image and M
data_dir = '../data/'
img1 = plt.imread(data_dir + 'im1.png')
M = np.max(img1.shape)

# Load the correspondences
corresp = np.load(data_dir + 'some_corresp.npz')
pts1 = corresp['pts1']
pts2 = corresp['pts2']

# Compute the Fundamental Matrix 
F = submission.eightpoint(pts1, pts2, M)

# Load the Intrinsic Matrices
intrinsics = np.load(data_dir + 'intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']

# Compute the Essential Matrix
E = submission.essentialMatrix(F, K1, K2)

# Compute M1, C1, and M2s
M1 = np.concatenate((np.eye(3), np.zeros((3, 1))), axis = 1)
C1 = K1 @ M1
M2s = helper.camera2(E)

# Find the correct M2
for i in range(4):
    # Check each M2
    M2 = M2s[:, :, i]
    C2 = K2 @ M2
    P, err = submission.triangulate(C1, pts1, C2, pts2)
    
    # Check the validity (z is positive)
    if np.min(P[:, 2]) > 0:
        # Print the reprojection error 
        print(f"reprojection error = {err}")
        break

# Save M2, C2, and P
results_dir = '../results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
np.savez(results_dir+ 'q3_3.npz', M2 = M2, C2 = C2, P = P) 