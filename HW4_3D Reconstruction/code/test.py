import numpy as np
import submission

# a = np.array([1, 2, 3])
# print(a.reshape([-1]))
# b = np.array([[1], [2], [3]])
# print(b.reshape([-1]))

data_dir = '../data/'
results_dir = '../results/'

# Check Q2.1
print("\nQ2.1")
Q2_1 = np.load(results_dir + 'q2_1.npz')
F = Q2_1['F']
M = Q2_1['M']
print(f"F = {F}\nM = {M}")

# Check Q3.1
print("\nQ3.1")
# Load the Intrinsic Matrix 
intrinsics = np.load(data_dir + 'intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']
# Compute the Essential Matrix
E = submission.essentialMatrix(F, K1, K2)
print(f"E = {E}") 

# Check Q3.3
print("\nQ3.3")
Q3_3 = np.load(results_dir + 'q3_3.npz')
M2 = Q3_3['M2']
C2 = Q3_3['C2']
P = Q3_3['P']
print(f"M2 = {M2}\nC2 = {C2}\nP = {P}, {P.shape}")

# Check Q4.1
print("\nQ4.1")
Q4_1 = np.load(results_dir + 'q4_1.npz')
F = Q4_1['F']
pts1 = Q4_1['pts1']
pts2 = Q4_1['pts2']
print(f"F = {F}\npts1 = {pts1}\npts2 = {pts2}")

# Check Q4.2
print("\nQ4.2")
Q4_2 = np.load(results_dir + 'q4_2.npz')
F = Q4_2['F']
M1 = Q4_2['M1']
M2 = Q4_2['M2']
C1 = Q4_2['C1']
C2 = Q4_2['C2']
print(f"F = {F}\nM1 = {M1}\nM2 = {M2}\nC1 = {C1}\nC2 = {C2}") 