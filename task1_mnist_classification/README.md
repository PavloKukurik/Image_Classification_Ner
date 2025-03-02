# MNIST Classification (Task 1)

This task implements three different models for handwritten digit classification using the MNIST dataset. The models are designed in an Object-Oriented Programming (OOP) style, ensuring modularity and reusability
## Task Overview
The task requires implementing the following models:
- **Random Forest Classifier (`rf`)** – A traditional machine learning model using Scikit-Learn.
- **Feed-Forward Neural Network (`nn`)** – A basic multi-layer perceptron (MLP) implemented with PyTorch.
- **Convolutional Neural Network (`cnn`)** – A deep learning model leveraging convolutional layers for image feature extraction.

Each model is implemented as a separate class following an interface called `MnistClassifierInterface`, which enforces two methods: `train()` and `predict()`. A wrapper class `MnistClassifier` is designed to allow seamless model selection.

## Project Structure
```
task1_mnist_classification/
│── mnist_classifier/
│   ├── cnn_classifier.py            # CNN Classifier
│   ├── feedforward_nn_classifier.py  # FFNN Classifier
│   ├── mnist_classifier.py            # Wrapper Class
│   ├── mnist_classifier_interface.py  # Interface
│   ├── random_forest_classifier.py    # RF Classifier
│   ├── train.py                        # Training Script
│   ├── task_1.ipynb                    # Jupyter Notebook with analysis
│── README.md                           # Documentation
│── requirements.txt                     # Dependencies
```

## Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/PavloKukurik/Test_task_Winstars
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Train and Test Models


### Jupyter Notebook
A Jupyter Notebook (`task_1.ipynb`) is included to demonstrate the workflow and provide example predictions, including edge cases. Run it to see the result



## Dependencies
The project relies on the following libraries:
```
numpy
pandas
matplotlib
scikit-learn
torch
torchvision
jupyter
```
