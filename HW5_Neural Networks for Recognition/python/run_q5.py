import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1.1 & Q5.1.2
# initialize layers here
##########################
##### your code here #####
##########################
# Initialize input, hidden, and output layer
layer_names = ['input', 'hlayer1', 'hlayer2', 'output']  
initialize_weights(train_x.shape[1], hidden_size, params, layer_names[0])
initialize_weights(hidden_size, hidden_size, params, layer_names[1])
initialize_weights(hidden_size, hidden_size, params, layer_names[2])
initialize_weights(hidden_size, train_x.shape[1], params, layer_names[3])

# Initialize momentum accumulators with zeros
layer_names = ['input', 'hlayer1', 'hlayer2', 'output']
for layer in layer_names:
    params['m_W' + layer] = np.zeros_like(params['W' + layer])
    params['m_b' + layer] = np.zeros_like(params['b' + layer])

# Initialize training loss list
list_of_train_loss_per_iter = []

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################
        # Forward propagation
        h1 = forward(xb, params, layer_names[0], relu)
        h2 = forward(h1, params, layer_names[1], relu)
        h3 = forward(h2, params, layer_names[2], relu)
        probs = forward(h3, params, layer_names[3], sigmoid)

        # Compute and Update loss
        loss = np.sum((probs - xb) ** 2)
        total_loss += loss

        # Back propagation
        # delta1 = derivative of (x-y)^2
        delta1 = 2 * (probs - xb)
        delta2 = backwards(delta1, params, layer_names[3], sigmoid_deriv)
        delta3 = backwards(delta2, params, layer_names[2], relu_deriv)
        delta4 = backwards(delta3, params, layer_names[1], relu_deriv)
        backwards(delta4, params, layer_names[0], relu_deriv)

        # Apply gradient descent with momentum
        for layer in layer_names:
            # Update weights
            params['m_W' + layer] = 0.9 * params['m_W' + layer] - learning_rate * params['grad_W' + layer]
            params['W' +  layer] += params['m_W' +  layer]
            # Update biases
            params['m_b' + layer] = 0.9 * params['m_b' + layer] - learning_rate * params['grad_b' + layer]
            params['b' +  layer] += params['m_b' +  layer]

    # Update training loss
    list_of_train_loss_per_iter.append(total_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr, total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# Q5.2
import matplotlib.pyplot as plt 
# Plot the training process
plt.figure()
plt.plot(np.arange(1, max_iters+1), list_of_train_loss_per_iter, color = 'r')
plt.title(f"Training Process: loss v.s. epoch")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
        
# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
##########################
##### your code here #####
##########################
# Forward propagation for the validation data
h1 = forward(valid_x, params, layer_names[0], relu)
h2 = forward(h1, params, layer_names[1], relu)
h3 = forward(h2, params, layer_names[2], relu)
valid_probs = forward(h3, params, layer_names[3], sigmoid)

# Extract 5 classes from the total 36 classes
class_index = [1, 5, 10, 20, 30]
valid_y = valid_data['valid_labels']
extracted_classes, single_class = [], []
for index in class_index:
    for i in range(len(valid_y)):
        if np.argmax(valid_y, axis = 1)[i] == index:
            single_class.append(i)
    extracted_classes.append(single_class)
    single_class = []

# Visualize 2 validation and reconstructed images for each class
for i in range(len(extracted_classes)):
    for j in range(2):
        index = extracted_classes[i][j]
        # Plot the validation and reconstructed images 
        fig, [ax1, ax2] = plt.subplots(1, 2)
        ax1.imshow(valid_x[index].reshape(32,32).T)
        ax1.set_title("Validation Image")
        ax2.imshow(valid_probs[index].reshape(32,32).T)
        ax2.set_title("Reconstructed Image")
        plt.show()
         


# Q5.3.2
# skimage version == 0.18.1
from skimage.metrics import peak_signal_noise_ratio as psnr
# evaluate PSNR
##########################
##### your code here #####
##########################
# Forward propagation for the validation data
h1 = forward(valid_x, params, layer_names[0], relu)
h2 = forward(h1, params, layer_names[1], relu)
h3 = forward(h2, params, layer_names[2], relu)
valid_probs = forward(h3, params, layer_names[3], sigmoid)

# Compute the average psnr for all validation images
total_psnr = 0
for i in range(len(valid_x)):
    total_psnr += psnr(valid_x[i], valid_probs[i])
average_psnr = total_psnr / len(valid_x) 
print(f"Average PSNR = {average_psnr}")