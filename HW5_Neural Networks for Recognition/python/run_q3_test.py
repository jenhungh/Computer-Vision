import numpy as np
import scipy.io
from nn import *
import pickle
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50  # For q4, max_iters = 100
batch_size = 72
learning_rate = 4e-2  # For q4, lr = 5e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# We have initialized the single layer network for you here
# Do not change the layer names
initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
initialize_weights(hidden_size,train_y.shape[1],params,'output')

# Store the intialized weights for Q3.3
initial_w = np.copy(params['Wlayer1'])

list_of_train_acc_per_iter = []
list_of_val_acc_per_iter = []
for itr in range(max_iters):
    total_train_loss = 0
    total_train_acc = 0
    for xb,yb in batches:
        ##########################
        ##### your code here #####
        ##########################
        # Forward propagation
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        
        # Compute and Update loss and accuracy
        loss, acc = compute_loss_and_acc(yb, probs)
        total_train_loss += loss
        total_train_acc += acc

        # Back propagation
        # Derivative of cross-entropy(softmax(x))
        delta1 = probs - yb
        delta2 = backwards(delta1,params,'output',linear_deriv)
        # Implement back propagation
        backwards(delta2,params,'layer1',sigmoid_deriv)

        # Apply gradient descent
        # Ouput layer
        params['W' + 'output'] -= learning_rate * params['grad_W' + 'output']
        params['b' + 'output'] -= learning_rate * params['grad_b' + 'output']
        # Hidden layer
        params['W' + 'layer1'] -= learning_rate * params['grad_W' + 'layer1']
        params['b' + 'layer1'] -= learning_rate * params['grad_b' + 'layer1']
    
    # Average training loss and accuracy
    total_train_loss = total_train_loss / batch_num 
    total_train_acc = total_train_acc / batch_num 
    
    val_acc = None
    # compute the validation accuracy here, make sure there is little overfitting issue
    ##########################
    ##### your code here #####
    ##########################
    # Forward propagation
    h1 = forward(valid_x, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
        
    # Compute loss and accuracy
    val_loss, val_acc = compute_loss_and_acc(valid_y, probs)

    # Update training and validation accuracy
    list_of_train_acc_per_iter.append(total_train_acc)
    list_of_val_acc_per_iter.append(val_acc)

    print("itr: {:02d} \t train loss: {:.2f} \t train acc : {:.2f} \t validation acc : {:.2f}".format(itr+1,total_train_loss,total_train_acc, val_acc))

# In a single plot, plot the training v.s. validation accuracy per iteration
##########################
##### your code here #####
##########################
# Plot the training and validation process
plt.figure()
plt.plot(np.arange(1, max_iters+1), list_of_train_acc_per_iter, color = 'r', label = 'training')
plt.plot(np.arange(1, max_iters+1), list_of_val_acc_per_iter, color = 'b', label = 'validation')
plt.title(f"Training and Validation Process: accuracy v.s. epoch\nlr = {learning_rate}")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# run on test set and report accuracy! should be around 75%
test_acc = None
h1 = forward(test_x,params,'layer1')
probs = forward(h1,params,'output',softmax)
loss, test_acc = compute_loss_and_acc(test_y, probs)

print('Test accuracy: ',test_acc)
# saved_params = {k:v for k,v in params.items() if '_' not in k}
# with open('q3_weights.pickle', 'wb') as handle:
#     pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# The weights after training loop are visualized here. 
# You may use the same visualization script to visualize the layer right after Xavier initialization
fig = plt.figure(1, (8., 8.))
if hidden_size < 128:
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
    img_w = params['Wlayer1'].reshape((32,32,hidden_size))
    for i in range(hidden_size):
        grid[i].imshow(img_w[:,:,i])  # The AxesGrid object work as a list of axes.
    plt.show()

# Visualize the layer right after Xavier initialization
# Line 30-31 stores the Xavier Initialization
fig = plt.figure(1, (8., 8.))
if hidden_size < 128:
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
    initial_w = initial_w.reshape((32,32,hidden_size))
    for i in range(hidden_size):
        grid[i].imshow(initial_w[:,:,i])  # The AxesGrid object work as a list of axes.
    plt.show()


# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
for i in range(test_y.shape[0]):
    i1 = np.argmax(probs[i])
    i2 = np.argmax(test_y[i])
    confusion_matrix[i1,i2] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()