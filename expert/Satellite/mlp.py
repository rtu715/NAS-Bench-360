import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import torch.utils.data as data_utils
from torch.utils.data import Dataset

torch.manual_seed(1)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 46
hidden_size_1 = 40
hidden_size_2 = 30
num_classes = 24
num_epochs = 200
batch_size = 1024
learning_rate = 0.001


def load_satellite_data(path):
    train_file = os.path.join(path, 'satellite_train.npy')
    test_file = os.path.join(path, 'satellite_test.npy')

    all_train_data, all_train_labels = np.load(train_file, allow_pickle=True)[()]['data'], np.load(train_file,allow_pickle=True)[()]['label']
    test_data, test_labels = np.load(test_file, allow_pickle=True)[()]['data'], np.load(test_file, allow_pickle=True)[()]['label']

    #rerange labels to 0-23
    all_train_labels = all_train_labels - 1
    test_labels = test_labels - 1

    #normalize data
    all_train_data = (all_train_data - all_train_data.mean(axis=1, keepdims=True))/all_train_data.std(axis=1, keepdims=True)
    test_data = (test_data - test_data.mean(axis=1, keepdims=True))/test_data.std(axis=1, keepdims=True)

    #convert to tensor/longtensor
    all_train_tensors, all_train_labeltensor = torch.from_numpy(all_train_data).type(torch.FloatTensor), \
                                               torch.from_numpy(all_train_labels).type(torch.LongTensor)

    test_tensors, test_labeltensor = torch.from_numpy(test_data).type(torch.FloatTensor), torch.from_numpy(test_labels).type(torch.LongTensor)
    testset = data_utils.TensorDataset(test_tensors, test_labeltensor)

    trainset = data_utils.TensorDataset(all_train_tensors, all_train_labeltensor)

    return trainset, None, testset

train_dataset, _, test_dataset = load_satellite_data('.')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Fully connected neural network with one hidden layer


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


model = NeuralNet(input_size, hidden_size_1, hidden_size_2, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 46).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 46).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: {} %'.format(
        100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
