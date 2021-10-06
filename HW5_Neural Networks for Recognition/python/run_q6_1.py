# Import package
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from nn import get_random_batches

# Load the NIST36 dataset
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

# Turn the dataset (numpy) into tensor
train_x, train_y = torch.from_numpy(train_x).float(),  torch.from_numpy(train_y)
valid_x, valid_y = torch.from_numpy(valid_x).float(),  torch.from_numpy(valid_y)
test_x, test_y = torch.from_numpy(test_x).float(),  torch.from_numpy(test_y)

# Q6.1.1 Fully-connected Neural Network
# Set up the hyperparameters
max_iters = 100
batch_size = 72
learning_rate = 0.1
hidden_size = 64
batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches) 

# Build the fully-connected network
class FullyConnectedNetwork(nn.Module):
    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()
        self.fc_layer_set = nn.Sequential(
            nn.Linear(train_x.shape[1], hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, train_y.shape[1]),
        )

    def forward(self, x):
        # Fully-connected layer
        x = self.fc_layer_set(x)  
        return x

# Train the fully-connected layer
# Load the model and Set up the criterion and optimizer 
fcn_model = FullyConnectedNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(fcn_model.parameters(), lr = learning_rate)

# Initialize training/validation loss and accuracy list
list_of_train_loss_per_iter = []
list_of_train_acc_per_iter = []
list_of_valid_loss_per_iter = []
list_of_valid_acc_per_iter = []

# Train the model
for iter in range(max_iters):
    total_train_loss = 0
    total_train_acc = 0
    for xb, yb in batches:
        # Turn yb into labelb
        labelb = torch.argmax(yb, dim = 1) 

        # Forward propagation
        optimizer.zero_grad()
        probs = fcn_model(xb)

        # Back propagation + Optimize
        loss = criterion(probs, labelb)
        loss.backward()
        optimizer.step()

        # Compute loss and accuaracy
        total_train_loss += loss.item()
        _, pred_label = torch.max(probs.data, 1)
        acc = torch.eq(pred_label, labelb).float().sum()
        total_train_acc += acc.item()
    
    # Average and Update training loss and accuracy
    average_train_loss = total_train_loss / batch_num
    average_train_acc = total_train_acc / train_x.shape[0]
    list_of_train_loss_per_iter.append(average_train_loss)
    list_of_train_acc_per_iter.append(average_train_acc)
    
    # Validation process
    # Turn valid_y into valid_label
    valid_label = torch.argmax(valid_y, dim = 1)

    # Forward propagation
    valid_probs = fcn_model(valid_x)
    
    # Compute loss and accuracy
    valid_loss = criterion(valid_probs, valid_label) 
    _, pred_valid_label = torch.max(valid_probs.data, 1)
    valid_acc = torch.eq(pred_valid_label, valid_label).float().mean()

    # Update validation loss and accuracy
    list_of_valid_loss_per_iter.append(valid_loss)
    list_of_valid_acc_per_iter.append(valid_acc)

    # Print the training and validation process
    print("iter: {:02d}".format(iter+1))
    print("train loss: {:.2f} \t train acc : {:.2f}".format(average_train_loss, average_train_acc))
    print("validation loss : {:.2f} \t validation acc : {:.2f}".format(valid_loss, valid_acc))

# Test the model with the test dataset
# Turn test_y into test_label
test_label = torch.argmax(test_y, dim = 1)
# Forward propagation
test_probs = fcn_model(test_x)
# Compute loss and accuracy 
test_loss = criterion(test_probs, test_label) 
_, pred_test_label = torch.max(test_probs.data, 1)
test_acc = torch.eq(pred_test_label, test_label).float().mean() 
print("Test\ntest loss : {:.2f} \t test accuracy : {:.2f}".format(test_loss, test_acc))

# Plot the training and validation process (loss)
plt.figure()
plt.plot(np.arange(1, max_iters+1), list_of_train_loss_per_iter, color = 'r', label = 'training')
plt.plot(np.arange(1, max_iters+1), list_of_valid_loss_per_iter, color = 'b', label = 'validation')
plt.title(f"Training and Validation Process: loss v.s. epoch")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# Plot the training and validation process (accuracy)
plt.figure()
plt.plot(np.arange(1, max_iters+1), list_of_train_acc_per_iter, color = 'r', label = 'training')
plt.plot(np.arange(1, max_iters+1), list_of_valid_acc_per_iter, color = 'b', label = 'validation')
plt.title(f"Training and Validation Process: accuracy v.s. epoch")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Q6.1.2 Convolutional Neural Network
# Set up the hyperparameters
max_iters = 10
batch_size = 72
learning_rate = 4e-3
train_batches = get_random_batches(train_x, train_y, batch_size)
train_batch_num = len(train_batches)
valid_batches = get_random_batches(valid_x, valid_y, batch_size)
valid_batch_num = len(valid_batches)

# Build the CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional block 
        self. conv_layer_set = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p = 0.1),
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p = 0.1)
        )
        # Fully-connected block
        self. fc_layer_set = nn.Sequential(
            nn.Linear(32 * 8 * 8, 1024),
            nn.ReLU(inplace = True),
            nn.Linear(1024, 36),
        )

    def forward(self, x):
        # Convolution layer block
        x = self.conv_layer_set(x)
        x = x.view(-1, 32 * 8 * 8)
        # Fully-connected layer block
        x = self.fc_layer_set(x)
        return x

# Train the CNN
# Choose the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and Set up the criterion and optimizer 
cnn_model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr = learning_rate)

# Initialize training/validation loss and accuracy list
list_of_train_loss_per_iter = []
list_of_train_acc_per_iter = []
list_of_valid_loss_per_iter = []
list_of_valid_acc_per_iter = []

# Train the model
for iter in range(max_iters):
    total_train_loss = 0
    total_train_acc = 0
    total_valid_loss = 0
    total_valid_acc = 0
    for xb, yb in train_batches:
        # Reshape xb and Turn yb into labelb
        xb = xb.reshape(batch_size, 1, 32, 32).to(device)
        labelb = torch.argmax(yb, dim = 1).to(device) 

        # Forward propagation
        optimizer.zero_grad()
        probs = cnn_model(xb)

        # Back propagation + Optimize
        loss = criterion(probs, labelb)
        loss.backward()
        optimizer.step()

        # Compute loss and accuaracy
        total_train_loss += loss.item()
        _, pred_label = torch.max(probs.data, 1)
        acc = torch.eq(pred_label, labelb).float().sum()
        total_train_acc += acc.item()
    
    # Average and Update training loss and accuracy
    average_train_loss = total_train_loss / train_batch_num
    average_train_acc = total_train_acc / train_x.shape[0]
    list_of_train_loss_per_iter.append(average_train_loss)
    list_of_train_acc_per_iter.append(average_train_acc)
    
    # Validation process
    for valid_xb, valid_yb in valid_batches:
        # Reshape xb and Turn yb into labelb
        valid_xb = valid_xb.reshape(batch_size, 1, 32, 32).to(device)
        valid_labelb = torch.argmax(valid_yb, dim = 1).to(device) 

        # Forward propagation
        optimizer.zero_grad()
        valid_probs = cnn_model(valid_xb)

        # Back propagation + Optimize
        loss = criterion(valid_probs, valid_labelb)
        loss.backward()
        optimizer.step()

        # Compute loss and accuaracy
        total_valid_loss += loss.item()
        _, pred_valid_label = torch.max(valid_probs.data, 1)
        valid_acc = torch.eq(pred_valid_label, valid_labelb).float().sum()
        total_valid_acc += valid_acc.item()
    
    # Average and Update validation loss and accuracy
    average_valid_loss = total_valid_loss / valid_batch_num
    average_valid_acc = total_valid_acc / valid_x.shape[0]
    list_of_valid_loss_per_iter.append(average_valid_loss)
    list_of_valid_acc_per_iter.append(average_valid_acc)

    # Print the training and validation process
    print("iter: {:02d}".format(iter+1))
    print("train loss: {:.2f} \t train acc : {:.2f}".format(average_train_loss, average_train_acc))
    print("validation loss : {:.2f} \t validation acc : {:.2f}".format(average_valid_loss, average_valid_acc))

# Test the model with the test dataset
# Reshape test_x and Turn test_y into test_label
test_x = test_x.reshape(test_x.shape[0], 1, 32, 32).to(device)
test_label = torch.argmax(test_y, dim = 1).to(device)
# Forward propagation
test_probs = cnn_model(test_x)
# Compute loss and accuracy 
test_loss = criterion(test_probs, test_label) 
_, pred_test_label = torch.max(test_probs.data, 1)
test_acc = torch.eq(pred_test_label, test_label).float().mean() 
print("Test\ntest loss : {:.2f} \t test accuracy : {:.2f}".format(test_loss, test_acc))

# Plot the training and validation process (loss)
plt.figure()
plt.plot(np.arange(1, max_iters+1), list_of_train_loss_per_iter, color = 'r', label = 'training')
plt.plot(np.arange(1, max_iters+1), list_of_valid_loss_per_iter, color = 'b', label = 'validation')
plt.title(f"Training and Validation Process: loss v.s. epoch")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# Plot the training and validation process (accuracy)
plt.figure()
plt.plot(np.arange(1, max_iters+1), list_of_train_acc_per_iter, color = 'r', label = 'training')
plt.plot(np.arange(1, max_iters+1), list_of_valid_acc_per_iter, color = 'b', label = 'validation')
plt.title(f"Training and Validation Process: accuracy v.s. epoch")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Q6.1.3 Convolutional Neural Network on CIFAR-10 dataset
# Set up the hyperparameters
max_iters = 30
batch_size = 200
learning_rate = 0.001

# Load the training and testing datatset
# Set up the transform for training and testing dataset
transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# Load the dataset 
train_set = torchvision.datasets.CIFAR10(root = './data', train = True,
                                        download = True, transform = transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
test_set = torchvision.datasets.CIFAR10(root = './data', train = False,
                                        download = True, transform = transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)

# Build the CNN
class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        # Convolutional block 
        self. conv_layer_set = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p = 0.1)
        )
        # Fully-connected block
        self. fc_layer_set = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(inplace = True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace = True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        # Convolution layer block
        x = self.conv_layer_set(x)
        x = x.view(-1, 256 * 8 * 8)
        # Fully-connected layer block
        x = self.fc_layer_set(x)
        return x

# Train the CNN
# Choose the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and Set up the criterion and optimizer 
cnn_cifar10_model = CNN_CIFAR10().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_cifar10_model.parameters(), lr = learning_rate)

# Initialize training/validation loss and accuracy list
list_of_train_loss_per_iter = []
list_of_train_acc_per_iter = []
list_of_valid_loss_per_iter = []
list_of_valid_acc_per_iter = []

# Train the model
for iter in range(max_iters):
    total_train_loss = 0
    total_train_acc = 0
    total_valid_loss = 0
    total_valid_acc = 0
    for train_data in train_loader:
        # Extract xb and labelb
        xb, labelb = train_data
        xb, labelb = xb.to(device), labelb.to(device)

        # Forward propagation
        optimizer.zero_grad()
        probs = cnn_cifar10_model(xb)

        # Back propagation + Optimize
        loss = criterion(probs, labelb)
        loss.backward()
        optimizer.step()

        # Compute loss and accuaracy
        total_train_loss += loss.item()
        _, pred_label = torch.max(probs.data, 1)
        acc = torch.eq(pred_label, labelb).float().sum()
        total_train_acc += acc.item()
    
    # Average and Update training loss and accuracy
    average_train_loss = total_train_loss / len(train_loader)
    average_train_acc = total_train_acc / len(train_set)
    list_of_train_loss_per_iter.append(average_train_loss)
    list_of_train_acc_per_iter.append(average_train_acc)

    # Print the training process
    print("iter: {:02d}".format(iter+1))
    print("train loss: {:.2f} \t train acc : {:.2f}".format(average_train_loss, average_train_acc))

# Test the model with the test dataset
test_loss = 0
test_acc = 0
for test_data in test_loader:
        # Extract test_xb and test_labelb
        test_xb, test_labelb = test_data
        test_xb, test_labelb = test_xb.to(device), test_labelb.to(device)

        # Forward propagation
        optimizer.zero_grad()
        test_probs = cnn_cifar10_model(test_xb)
        
        # Compute and Update loss and accuracy 
        loss = criterion(test_probs, test_labelb)
        test_loss += loss.item() 
        _, pred_test_label = torch.max(test_probs.data, 1)
        acc = torch.eq(pred_test_label, test_labelb).float().sum()
        test_acc += acc.item()

# Print the test loss and accuracy
test_loss = test_loss / len(test_loader)
test_acc = test_acc / len(test_set)
print("Test\ntest loss : {:.2f} \t test accuracy : {:.2f}".format(test_loss, test_acc))

# Plot the training process (loss)
plt.figure()
plt.plot(np.arange(1, max_iters+1), list_of_train_loss_per_iter, color = 'r')
plt.title(f"Training Process: loss v.s. epoch")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# Plot the training and validation process (accuracy)
plt.figure()
plt.plot(np.arange(1, max_iters+1), list_of_train_acc_per_iter, color = 'r')
plt.title(f"Training Process: accuracy v.s. epoch")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# Q6.1.4 Convolutional Neural Network on SUN dataset
# Set up the hyperparameters
max_iters = 30
batch_size = 50
learning_rate = 0.001

# Load the SUN dataset
transform = transforms.Compose([transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.425, 0.425, 0.425), (0.225, 0.225, 0.225))])
sun_data = torchvision.datasets.ImageFolder(root = '../SUN', transform = transform)
# Split into training and testing dataset
train_set, test_set = torch.utils.data.random_split(sun_data, [1177, 400])
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle = False)

# Build the CNN
class CNN_SUN(nn.Module):
    def __init__(self):
        super(CNN_SUN, self).__init__()
        # Convolutional block 
        self. conv_layer_set = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Dropout(p = 0.1)
        )
        # Fully-connected block
        self. fc_layer_set = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(inplace = True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace = True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        # Convolution layer block
        x = self.conv_layer_set(x)
        x = x.view(-1, 128 * 8 * 8)
        # Fully-connected layer block
        x = self.fc_layer_set(x)
        return x

# Train the CNN
# Choose the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and Set up the criterion and optimizer 
cnn_sun_model = CNN_SUN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_sun_model.parameters(), lr = learning_rate)

# Initialize training/validation loss and accuracy list
list_of_train_loss_per_iter = []
list_of_train_acc_per_iter = []
list_of_valid_loss_per_iter = []
list_of_valid_acc_per_iter = []

# Train the model
for iter in range(max_iters):
    total_train_loss = 0
    total_train_acc = 0
    total_valid_loss = 0
    total_valid_acc = 0
    for train_data in train_loader:
        # Extract xb and labelb
        xb, labelb = train_data
        xb, labelb = xb.to(device), labelb.to(device)

        # Forward propagation
        optimizer.zero_grad()
        probs = cnn_sun_model(xb)

        # Back propagation + Optimize
        loss = criterion(probs, labelb)
        loss.backward()
        optimizer.step()

        # Compute loss and accuaracy
        total_train_loss += loss.item()
        _, pred_label = torch.max(probs.data, 1)
        acc = torch.eq(pred_label, labelb).float().sum()
        total_train_acc += acc.item()
    
    # Average and Update training loss and accuracy
    average_train_loss = total_train_loss / len(train_loader)
    average_train_acc = total_train_acc / len(train_set)
    list_of_train_loss_per_iter.append(average_train_loss)
    list_of_train_acc_per_iter.append(average_train_acc)

    # Print the training process
    print("iter: {:02d}".format(iter+1))
    print("train loss: {:.2f} \t train acc : {:.2f}".format(average_train_loss, average_train_acc))

# Test the model with the test dataset
test_loss = 0
test_acc = 0
for test_data in test_loader:
        # Extract test_xb and test_labelb
        test_xb, test_labelb = test_data
        test_xb, test_labelb = test_xb.to(device), test_labelb.to(device)

        # Forward propagation
        optimizer.zero_grad()
        test_probs = cnn_sun_model(test_xb)
        
        # Compute and Update loss and accuracy 
        loss = criterion(test_probs, test_labelb)
        test_loss += loss.item() 
        _, pred_test_label = torch.max(test_probs.data, 1)
        acc = torch.eq(pred_test_label, test_labelb).float().sum()
        test_acc += acc.item()

# Print the test loss and accuracy
test_loss = test_loss / len(test_loader)
test_acc = test_acc / len(test_set)
print("Test\ntest loss : {:.2f} \t test accuracy : {:.2f}".format(test_loss, test_acc))

# Plot the training process (loss)
plt.figure()
plt.plot(np.arange(1, max_iters+1), list_of_train_loss_per_iter, color = 'r')
plt.title(f"Training Process: loss v.s. epoch")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# Plot the training and validation process (accuracy)
plt.figure()
plt.plot(np.arange(1, max_iters+1), list_of_train_acc_per_iter, color = 'r')
plt.title(f"Training Process: accuracy v.s. epoch")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()