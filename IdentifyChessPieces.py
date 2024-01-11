import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from torchvision import transforms


def build_model():
    # Load data
    training_images = torch.from_numpy(np.load("training_images.npy")).float()
    training_labels = torch.from_numpy(np.load("training_labels.npy")).long()
    testing_images = torch.from_numpy(np.load("testing_images.npy")).float()
    testing_labels = torch.from_numpy(np.load("testing_labels.npy")).long()

    # Normalize pixel values to be between 0 and 1
    training_images, testing_images = training_images / 255.0, testing_images / 255.0

    # Convert labels to one-hot encoding
    training_labels = torch.nn.functional.one_hot(training_labels)
    testing_labels = torch.nn.functional.one_hot(testing_labels)

    # Create an instance of the model
    model = CNNModel()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert data to DataLoader
    train_dataset = TensorDataset(training_images, training_labels.argmax(dim=1))
    test_dataset = TensorDataset(testing_images, testing_labels.argmax(dim=1))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Training the model
    epochs = 10
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print("Accuracy: {}".format(accuracy))

    # Save the model
    torch.save(model.state_dict(), "chess_piece_classifier.pth")


# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(96 * 10 * 10, 104)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(104, 13)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x