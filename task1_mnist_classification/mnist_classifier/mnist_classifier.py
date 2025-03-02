"""
Class wrapper for all models
"""

from cnn_classifier import CnnMnistClassifier
from mnist_classifier_interface import MnistClassifierInterface
from random_forest_classifier import RandomForestMnistClassifier
from feedforward_nn_classifier import FeedForwardMnistClassifier


class MnistClassifier(MnistClassifierInterface):
    """
    Wrapper class for different MNIST classifiers.
    This class takes the name of the algorithm ('rf', 'nn', 'cnn') and
    initializes the corresponding model.

    Supported model types:
    - 'rf': Random Forest classifier
    - 'nn': Feedforward Neural Network classifier
    - 'cnn': Convolutional Neural Network classifier
    """

    def __init__(self, model_type: str):
        """
        Initializes the MnistClassifier with the specified model type.
        :param model_type: A string indicating the type of model to use. Must be one of ['rf', 'nn', 'cnn'].
        :raises ValueError: If an invalid model type is provided.
        """
        if model_type == "rf":
            self.model = RandomForestMnistClassifier()
        elif model_type == "nn":
            self.model = FeedForwardMnistClassifier()
        elif model_type == "cnn":
            self.model = CnnMnistClassifier()
        else:
            raise ValueError("Invalid model type. Choose from ['rf', 'nn', 'cnn']")

    def train(self, X_train, y_train):
        """
        Trains the initialized model using the provided training data.
        :param X_train: A numpy array of shape (n_samples, n_features) containing the input features for training.
        :param y_train: A numpy array of shape (n_samples,) containing the corresponding labels for the training data.
        :return: None
        """
        self.model.train(X_train, y_train)

    def predict(self, X):
        """
        Predicts class labels for the given input data using the trained model.
        :param X: A numpy array of shape (n_samples, n_features) containing the input data for which to predict labels.
        :return: A numpy array of shape (n_samples,) containing the predicted class labels.
        """
        return self.model.predict(X)
