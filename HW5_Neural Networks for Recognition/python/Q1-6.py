import numpy as np
import matplotlib.pyplot as plt

# Sigmoid
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

x = np.linspace(-10, 10, 100)
s = sigmoid(x)

# Plot the function
plt.plot(x, s)
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.show()

# Derivative of sigmoid
def dsigmoid(x):
    ds = (1 - sigmoid(x)) * sigmoid(x)
    return ds

ds = dsigmoid(x)

# Plot the derivaitve of the function
plt.plot(x, ds)
plt.title("Derivative of the Sigmoid Function")
plt.xlabel("x")
plt.ylabel("dSigmoid(x)/dx")
plt.show()

# Tanh
def tanh(x):
    t = (1 - np.exp(-2*x))/ (1 + np.exp(-2*x))
    return t

x = np.linspace(-10, 10, 100)
t = tanh(x)

# Plot the function
plt.plot(x, t)
plt.title("Tanh Function")
plt.xlabel("x")
plt.ylabel("Tanh(x)")
plt.show()

# Derivative of sigmoid
def dtanh(x):
    dt = 1 - (tanh(x) ** 2)
    return dt

dt = dtanh(x)

# Plot the derivaitve of the function
plt.plot(x, dt)
plt.title("Derivative of the Tanh Function")
plt.xlabel("x")
plt.ylabel("dTanh(x)/dx")
plt.show()

# ReLU
def ReLU(x):
    r = max(0, x)
    return r

x = np.linspace(-10, 10, 100)
r = [ReLU(i) for i in x] 

# Plot the function
plt.plot(x, r)
plt.title("ReLU Function")
plt.xlabel("x")
plt.ylabel("ReLU(x)")
plt.show()

# Derivative of sigmoid
def dReLU(x):
    if x > 0:
        dr = 1
    if x <= 0:
        dr = 0
    return dr

dr = [dReLU(i) for i in x]

# Plot the derivaitve of the function
plt.plot(x, dr)
plt.title("Derivative of the ReLU Function")
plt.xlabel("x")
plt.ylabel("dReLU(x)/dx")
plt.show()