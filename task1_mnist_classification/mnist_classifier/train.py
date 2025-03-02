"""
Python Scrypt for testing models on MNIST dataset
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from mnist_classifier import MnistClassifier
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target.astype(np.int32)


# (Models: 'rf', 'nn', 'cnn')
model_type = 'cnn'
classifier = MnistClassifier(model_type)

if model_type == 'cnn':
    # For CNN
    X, y = X[:20000], y[:20000]

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


classifier.train(X_train, y_train)

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy for {model_type}: {acc:.4f}")
