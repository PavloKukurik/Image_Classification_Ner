"""
Implementation of Feed-Forward Neural Network for MNIST Classification.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from mnist_classifier_interface import MnistClassifierInterface
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
EPOCHS = 12
LEARNING_RATE = 0.01


class FeedForwardNN(nn.Module):
    """
    A simple Feed-Forward Neural Network for MNIST classification.
    This model consists of an input layer, one hidden layer with ReLU activation,
    and an output layer with 10 neurons corresponding to the 10 digit classes.
    """

    def __init__(self):
        """
        Initializes the FeedForwardNN model.
        """
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.
        :param x: A tensor of shape (batch_size, INPUT_SIZE)
        :return: A tensor of shape (batch_size, OUTPUT_SIZE) containing the class scores
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class FeedForwardMnistClassifier(MnistClassifierInterface):
    """
    Feed-Forward Neural Network-based classifier for MNIST.
    This classifier uses a fully connected neural network with one hidden layer
    and ReLU activation to classify MNIST digits.
    """

    def __init__(self):
        """
        Initializes the FeedForwardMnistClassifier.
        Sets up the model, loss function (CrossEntropyLoss), and optimizer (Adam).
        """
        self.model = FeedForwardNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def train(self, X_train: np.ndarray | pd.DataFrame, y_train: np.ndarray):
        """
        Trains the Feed-Forward Neural Network using the provided training data.
        :param X_train: A numpy array of shape (n_samples, INPUT_SIZE)
        :param y_train: A numpy array of shape (n_samples,) containing class labels (0-9)
        :return: None
        """
        X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
        y_train = torch.tensor(np.array(y_train), dtype=torch.long)

        for epoch in range(EPOCHS):
            self.optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 2 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}")

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predicts class labels for the given input data.
        :param X: A numpy array of shape (n_samples, INPUT_SIZE)
        :return: A numpy array of shape (n_samples,) containing predicted class labels (0-9)
        """
        X = torch.tensor(np.array(X), dtype=torch.float32)
        outputs = self.model(X)
        _, predicted = torch.max(outputs, 1)
        return predicted.numpy()


if __name__ == "__main__":
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data.to_numpy(), mnist.target.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ffnn_classifier = FeedForwardMnistClassifier()
    ffnn_classifier.train(X_train, y_train)

    y_pred = ffnn_classifier.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"Feed-Forward NN Accuracy: {acc:.4f}")
