import numpy as np
import matplotlib.pyplot as plt
import helper
import submission
import os

# Q2.1
# Load the images
data_dir = '../data/'
img1 = plt.imread(data_dir + 'im1.png')
img2 = plt.imread(data_dir + 'im2.png')
M = np.max(img1.shape)

# Load the correspondences
corresp = np.load(data_dir + 'some_corresp.npz')
pts1 = corresp['pts1']
pts2 = corresp['pts2']

# Compute F 
F = submission.eightpoint(pts1, pts2, M)

# Save the Fundamental Matrix and scale
results_dir = '../results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
np.savez(results_dir+ 'q2_1.npz', F = F, M = M) 

# Check the .npz file
results_dir = '../results/'
Q2_1 = np.load(results_dir + 'q2_1.npz')
F = Q2_1['F']
M = Q2_1['M']
print(f"F = {F}\nM = {M}")

# Display epipolar lines
# helper.displayEpipolarF(img1, img2, F)

# Q3.1
# Load the Intrinsic Matrix 
intrinsics = np.load(data_dir + 'intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']

# Compute the Essential Matrix
E = submission.essentialMatrix(F, K1, K2)
print(f"E = {E}")

# Q3.2
# Compute M1 nand M2
M1 = np.concatenate((np.eye(3), np.zeros((3, 1))), axis = 1)
M2s = helper.camera2(E)
M21 = M2s[:, :, 0]
M22 = M2s[:, :, 1]
M23 = M2s[:, :, 2]
M24 = M2s[:, :, 3]

# Compute C1 and C2
C1 = K1 @ M1
C21 = K2 @ M21
C22 = K2 @ M22
C23 = K2 @ M23
C24 = K2 @ M24

# Traingulation
P, err1 = submission.triangulate(C1, pts1, C21, pts2)
P, err2 = submission.triangulate(C1, pts1, C22, pts2)
P, err3 = submission.triangulate(C1, pts1, C23, pts2)
P, err4 = submission.triangulate(C1, pts1, C24, pts2)
print(err1, err2, err3, err4)

# Find the correct M2
for i in range(4):
    # Check each M2
    M2 = M2s[:, :, i]
    C2 = K2 @ M2
    P, err = submission.triangulate(C1, pts1, C2, pts2)
    
    # Checkt the validity
    print(np.min(P[:, 2]))

# Check Q3_3.npz
results_dir = '../results/'
Q3_3 = np.load(results_dir + 'q3_3.npz')
M2 = Q3_3['M2']
C2 = Q3_3['C2']
P = Q3_3['P']
print(f"M2 = {M2}\nC2 = {C2}\nP = {P}, {P.shape}")

# W, err = submission.triangulate(C1, pts1, C2, pts2)
# print(W, err)

# Q4.1
helper.epipolarMatchGUI(img1, img2, F)
# Check
results_dir = '../results/'
Q4_1 = np.load(results_dir + 'q4_1.npz')
F = Q4_1['F']
pts1 = Q4_1['pts1']
pts2 = Q4_1['pts2']
print(f"F = {F}\npts1 = {pts1}\npts2 = {pts2}")

# Q4.2
# Check
results_dir = '../results/'
Q4_2 = np.load(results_dir + 'q4_2.npz')
F = Q4_2['F']
M1 = Q4_2['M1']
M2 = Q4_2['M2']
C1 = Q4_2['C1']
C2 = Q4_2['C2']
print(f"F = {F}\nM1 = {M1}\nM2 = {M2}\nC1 = {C1}\nC2 = {C2}")

# Q5.1
corresp_noise = np.load(data_dir + 'some_corresp_noisy.npz')
pts1_noise = corresp_noise['pts1']
pts2_noise = corresp_noise['pts2']
# F, inliers = submission.ransacF(pts1_noise, pts2_noise, M, nIters=1000, tol=0.42)
# print(f"F = {F}")
# print(f"inliers = {inliers}, num_inliers = {np.sum(inliers)}")

# Display epipolar lines
# helper.displayEpipolarF(img1, img2, F)

# Compare with Eight Point Algorithm
F8 = submission.eightpoint(pts1_noise, pts2_noise, M)
# helper.displayEpipolarF(img1, img2, F8)

# Q5.2
r = np.array([[1], [2], [3]])
R = submission.rodrigues(r)
print(f"R = {R}")

# R = np.eye(3)
r = submission.invRodrigues(R)
print(f"r = {r}")