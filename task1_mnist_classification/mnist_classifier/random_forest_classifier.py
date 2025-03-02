"""
Implementation of Random Forest Classifier for MNIST.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mnist_classifier_interface import MnistClassifierInterface
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class RandomForestMnistClassifier(MnistClassifierInterface):
    """
    Random Forest-based classifier for MNIST.
    This classifier normalizes input data using StandardScaler and
    utilizes a Random Forest model for classification.
    """

    def __init__(self, n_estimators=100):
        """
        Initializes the RandomForestMnistClassifier.
        :param n_estimators: The number of trees in the forest.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        self.scaler = StandardScaler()

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the Random Forest model using the provided training data.
        The input data is normalized before fitting the model.
        :param X_train: A numpy array of shape (n_samples, n_features)
        :param y_train: A numpy array of shape (n_samples,) containing
        :return: None
        """
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the given input data.
        The input data is normalized using the scaler fitted on
        the training data.
        :param X: A numpy array of shape (n_samples, n_features)
        :return: A numpy array of shape (n_samples,) containing
        """
        X = self.scaler.transform(X)
        return self.model.predict(X)


if __name__ == "__main__":
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestMnistClassifier()
    rf_classifier.train(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {acc:.4f}")
