import numpy as np
import matplotlib.pyplot as plt
import helper
import submission
import os

# Load the image and Compute M 
data_dir = '../data/'
img1 = plt.imread(data_dir + 'im1.png')
img2 = plt.imread(data_dir + 'im2.png')
M = np.max(img1.shape)

# Load the corrsponding points
corresp_noise = np.load(data_dir + 'some_corresp_noisy.npz')
pts1_noise = corresp_noise['pts1']
pts2_noise = corresp_noise['pts2']

# Compute F using RANSAC
F, inliers = submission.ransacF(pts1_noise, pts2_noise, M, nIters=1000, tol=0.42)
# Save F and inliers
results_dir = '../results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
np.savez(results_dir+ 'q5_3.npz', F = F, inliers = inliers) 

# Keep inliers only
pts1 = pts1_noise[inliers]
pts2 = pts2_noise[inliers]

# Load the Intrinsic Matrix 
intrinsics = np.load(data_dir + 'intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']

# Compute E 
E = submission.essentialMatrix(F, K1, K2)

# Compute C1 and C2
M1 = np.concatenate((np.eye(3), np.zeros((3, 1))), axis = 1) 
C1 = K1 @ M1
M2s = helper.camera2(E)
# Find the correct M2
for i in range(4):
    # Check each M2
    M2_init = M2s[:, :, i]
    C2 = K2 @ M2_init
    P_init, err = submission.triangulate(C1, pts1, C2, pts2)
    
    # Check the validity (z is positive)
    if np.min(P_init[:, 2]) > 0:
        # Print the reprojection error 
        print(f"reprojection error for M2_init = {err}")
        break

# Bundle Adjustment
M2, w = submission.bundleAdjustment(K1, M1, pts1, K2, M2_init, pts2, P_init)

# Compute reprojection errors
C2 = K2 @ M2
P, err_ba = submission.triangulate(C1=C1, pts1=pts1, C2=C2, pts2=pts2)
print(f"reprojection error for ba = {err_ba}")

# Plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection = '3d')
ax.scatter(P_init[:, 0], P_init[:, 1], P_init[:, 2], c = 'b', s = 3)
ax.set_title(f"Visualization of the Original 3D points\nreprojection error = {err}")
plt.setp(ax, xlabel = 'X', ylabel = 'Y', zlabel = 'Z')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection = '3d')
ax.scatter(P[:, 0], P[:, 1], P[:, 2], c = 'b', s = 3)
ax.set_title(f"Visualization of the Optimized 3D points\nreprojection error = {err_ba}")
plt.setp(ax, xlabel = 'X', ylabel = 'Y', zlabel = 'Z')
plt.show()