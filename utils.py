import numpy as np
from sklearn.preprocessing import OneHotEncoder


def encode_labels(y):
    encoder = OneHotEncoder(sparse=False)
    return encoder.fit_transform(y)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))
