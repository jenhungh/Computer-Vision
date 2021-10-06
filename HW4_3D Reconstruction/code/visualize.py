'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
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
img2 = plt.imread(data_dir + 'im2.png')
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

# Load the Temple Coordinates on image1 
templeCoords = np.load(data_dir + 'templeCoords.npz')
x1s = templeCoords['x1']
y1s = templeCoords['y1']
p1 = np.concatenate((x1s, y1s), axis = 1)

# Find the epipolar correspondence
p2 = np.zeros_like(p1)
for i in range(p2.shape[0]):
    x1, y1 = p1[i, 0], p1[i, 1]
    p2[i, 0], p2[i, 1] = submission.epipolarCorrespondence(img1, img2, F, x1, y1) 

# Find the correct M2
for i in range(4):
    # Check each M2
    M2 = M2s[:, :, i]
    C2 = K2 @ M2
    P, err = submission.triangulate(C1, p1, C2, p2)
    
    # Check the validity (z is positive)
    if np.min(P[:, 2]) > 0:
        # Print the reprojection error 
        print(f"reprojection error = {err}")
        break

# Plot the 3D reconstruction
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection = '3d')
ax.scatter(P[:, 0], P[:, 1], P[:, 2], c = 'b', s = 3)
ax.set_title('Visualization of 3D Recontruction')
plt.setp(ax, xlabel = 'X', ylabel = 'Y', zlabel = 'Z')
plt.show()

# Save F, M1, M2, C1, and C2
results_dir = '../results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
np.savez(results_dir+ 'q4_2.npz', F = F, M1 = M1, M2 = M2, C1 = C1, C2 = C2) 