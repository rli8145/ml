import numpy as np

def sigmoid(z):
    return np.round(1/(1 + np.exp(-z)), 5)

def relu(z):
    return np.maximum(0, z)