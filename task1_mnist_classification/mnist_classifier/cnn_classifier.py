"""
Implementation of Convolutional Neural Network (CNN) for MNIST classification.
"""

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from mnist_classifier_interface import MnistClassifierInterface

EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001


class CNN(nn.Module):
    """
    Convolutional Neural Network architecture for MNIST classification.
    The model consists of two convolutional layers followed by max pooling,
    and two fully connected layers with ReLU activations.
    """

    def __init__(self):
        """
        Initializes the CNN architecture.
        - Conv Layer 1: 1 input channel, 32 output channels, kernel size 3x3
        - Conv Layer 2: 32 input channels, 64 output channels, kernel size 3x3
        - Fully Connected Layer 1: 64*7*7 input features, 128 output features
        - Fully Connected Layer 2: 128 input features, 10 output features (for 10 classes)
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Forward pass through the network.
        :param x: Input tensor of shape (batch_size, 1, 28, 28)
        :return: LogSoftmax probabilities for each class.
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


class CnnMnistClassifier(MnistClassifierInterface):
    """
    CNN-based classifier for MNIST.
    This classifier normalizes input data and utilizes a Convolutional Neural
    Network for classification.
    """

    def __init__(self):
        """
        Initializes the CnnMnistClassifier.
        - Uses CrossEntropyLoss as the loss function.
        - Uses Adam optimizer with a learning rate of 0.001.
        - Automatically utilizes GPU if available.
        """
        self.model = CNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the CNN model using the provided training data.
        The input data is reshaped and moved to the GPU (if available).
        :param X_train: A numpy array of shape (n_samples, 28, 28)
        :param y_train: A numpy array of shape (n_samples,) containing class labels
        :return: None
        """

        self.model.train()

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28)
        y_train = torch.tensor(y_train, dtype=torch.long)

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss / len(dataloader):.4f}")

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predicts class labels for the given input data.
        The input data is reshaped and moved to the GPU (if available).
        :param X: A numpy array of shape (n_samples, 28, 28)
        :return: A numpy array of shape (n_samples,) containing class predictions
        """

        self.model.eval()
        X = torch.tensor(np.array(X), dtype=torch.float32).view(-1, 1, 28, 28).to(self.device)
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()


if __name__ == "__main__":
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data.to_numpy(), mnist.target.astype(np.int32)

    X = X / 255.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cnn_classifier = CnnMnistClassifier()
    cnn_classifier.train(X_train, y_train)

    y_pred = cnn_classifier.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"CNN Accuracy: {acc:.4f}")
