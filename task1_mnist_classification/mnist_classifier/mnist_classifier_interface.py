"""
Creating Interface for MNIST classifiers
"""


import numpy as np
from abc import ABC, abstractmethod


class MnistClassifierInterface(ABC):
    """
    Interface for MNIST classifiers.
    All models implementing this interface must provide
    implementations for the `train()` and `predict()` methods.
    """

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the model using the provided training data.
        :param X_train: A numpy array of shape (n_samples, n_features) containing the input features for training.
        :param y_train: A numpy array of shape (n_samples,) containing the corresponding labels for the training data.
        :return: None
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class labels for the given input data.
        :param X: A numpy array of shape (n_samples, n_features) containing the input data for which to predict labels.
        :return: A numpy array of shape (n_samples,) containing the predicted class labels.
        """
        pass
