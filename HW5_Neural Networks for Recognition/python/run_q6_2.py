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


# Fine-tune a single layer classifier with SqueezeNet
# Set up the hyperparameters
batch_size = 32
max_iters1 = 20
max_iters2 = 20
learning_rate1 = 0.001
learning_rate2 = 0.0001

# Load flowers17 dataset
# For fine tuning 
finetune_transform = transforms.Compose([transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.425, 0.425, 0.425), (0.225, 0.225, 0.225))])
finetune_train_set = torchvision.datasets.ImageFolder(root = '../data/oxford-flowers17/train', transform = finetune_transform)
finetune_test_set = torchvision.datasets.ImageFolder(root = '../data/oxford-flowers17/test', transform = finetune_transform)
# Load the dataset into DataLoader
finetune_train_loader = torch.utils.data.DataLoader(finetune_train_set, batch_size = batch_size, shuffle = True)
finetune_test_loader = torch.utils.data.DataLoader(finetune_test_set, batch_size = batch_size, shuffle = False)

# Load the SqueezeNet
squeeze_model = torchvision.models.squeezenet1_1(pretrained = True)
# Replace the classifier layer into 102 classes
squeeze_model.classifier[1] = nn.Conv2d(in_channels = 512, out_channels = 102, kernel_size = 1)

# Train the SqueezeNet : 2 Steps
# Choose the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the SqueezeNet
squeeze_model = torchvision.models.squeezenet1_1(pretrained = True).to(device)
# Replace the classifier layer into 102 classes
squeeze_model.classifier[1] = nn.Conv2d(in_channels = 512, out_channels = 102, kernel_size = 1).to(device)

# 1st Step : train the sinlgle classifier layer
for param in squeeze_model.parameters():
    param.requires_grad = False
for param in squeeze_model.classifier.parameters():
    param.requires_grad = True

# Set up the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(squeeze_model.classifier.parameters(), lr = learning_rate1)

# Initialize training loss and accuracy list
list_of_train_loss_per_iter = []
list_of_train_acc_per_iter = []

# Train the classifier layer
print("Train the single layer classifier")
for iter in range(max_iters1):
    total_train_loss = 0
    total_train_acc = 0
    for train_data in finetune_train_loader:
        # Extract xb and labelb
        xb, labelb = train_data
        xb, labelb = xb.to(device), labelb.to(device)

        # Forward propagation
        optimizer.zero_grad()
        probs = squeeze_model(xb)

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
    average_train_loss = total_train_loss / len(finetune_train_loader)
    average_train_acc = total_train_acc / len(finetune_train_set)
    list_of_train_loss_per_iter.append(average_train_loss)
    list_of_train_acc_per_iter.append(average_train_acc)

    # Print the training process
    print("iter: {:02d}".format(iter+1))
    print("train loss: {:.2f} \t train acc : {:.2f}".format(average_train_loss, average_train_acc))

# 2nd Step : finetune the SqueezeNet 
for param in squeeze_model.parameters():
    param.requires_grad = True

# Update the optimizer
optimizer = optim.Adam(squeeze_model.parameters(), lr = learning_rate2)

# Train the SqueezeNet
print("Train the SqueezeNet")
for iter in range(max_iters2):
    total_train_loss = 0
    total_train_acc = 0
    for train_data in finetune_train_loader:
        # Extract xb and labelb
        xb, labelb = train_data
        xb, labelb = xb.to(device), labelb.to(device)

        # Forward propagation
        optimizer.zero_grad()
        probs = squeeze_model(xb)

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
    average_train_loss = total_train_loss / len(finetune_train_loader)
    average_train_acc = total_train_acc / len(finetune_train_set)
    list_of_train_loss_per_iter.append(average_train_loss)
    list_of_train_acc_per_iter.append(average_train_acc)

    # Print the training process
    print("iter: {:02d}".format(iter+1))
    print("train loss: {:.2f} \t train acc : {:.2f}".format(average_train_loss, average_train_acc))

# Test the model with the test dataset
test_loss = 0
test_acc = 0
for test_data in finetune_test_loader:
        # Extract test_xb and test_labelb
        test_xb, test_labelb = test_data
        test_xb, test_labelb = test_xb.to(device), test_labelb.to(device)

        # Forward propagation
        optimizer.zero_grad()
        test_probs = squeeze_model(test_xb)
        
        # Compute and Update loss and accuracy 
        loss = criterion(test_probs, test_labelb)
        test_loss += loss.item() 
        _, pred_test_label = torch.max(test_probs.data, 1)
        acc = torch.eq(pred_test_label, test_labelb).float().sum()
        test_acc += acc.item()

# Print the test loss and accuracy
test_loss = test_loss / len(finetune_test_loader)
test_acc = test_acc / len(finetune_test_set)
print("Test\ntest loss : {:.2f} \t test accuracy : {:.2f}".format(test_loss, test_acc))

# Plot the training process (loss)
plt.figure()
plt.plot(np.arange(1, max_iters1+max_iters2+1), list_of_train_loss_per_iter, color = 'r')
plt.title(f"Training Process: loss v.s. epoch")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# Plot the training and validation process (accuracy)
plt.figure()
plt.plot(np.arange(1, max_iters1+max_iters2+1), list_of_train_acc_per_iter, color = 'r')
plt.title(f"Training Process: accuracy v.s. epoch")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# Self-defined CNN
# Set up the hyperparameters
batch_size = 32
max_iters = 40
learning_rate = 0.001

# Load flowers17 dataset
# For self-defined CNN
transform = transforms.Compose([transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.425, 0.425, 0.425), (0.225, 0.225, 0.225))])
train_set = torchvision.datasets.ImageFolder(root = '../data/oxford-flowers17/train', transform = transform)
test_set = torchvision.datasets.ImageFolder(root = '../data/oxford-flowers17/test', transform = transform)
# Load the dataset into DataLoader
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)

# Build the self-defined CNN (LeNet)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional block : 3 Convolutional Layers
        self. conv_layer_set = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5, stride = 1),
            nn.ReLU(inplace = True),
        )
        # Fully-connected block : 2 FC Layers
        self. fc_layer_set = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace = True),
            nn.Linear(84, 102),
        )

    def forward(self, x):
        # Convolution layer block
        x = self.conv_layer_set(x)
        x = x.view(-1, 120)
        # Fully-connected layer block
        x = self.fc_layer_set(x)
        return x

# Train the self-defined CNN
# Choose the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and Set up the criterion and optimizer 
cnn_model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr = learning_rate)

# Initialize training loss and accuracy list
list_of_train_loss_per_iter = []
list_of_train_acc_per_iter = []

# Train the model
for iter in range(max_iters):
    total_train_loss = 0
    total_train_acc = 0
    for train_data in train_loader:
        # Extract xb and labelb
        xb, labelb = train_data
        xb, labelb = xb.to(device), labelb.to(device)

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
        test_probs = cnn_model(test_xb)
        
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