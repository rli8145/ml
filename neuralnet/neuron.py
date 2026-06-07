import numpy as np
from numpy.typing import NDArray

class Neuron:
    def __init__(self, weights: NDArray[np.float64], bias: float, activation: str):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def forward(self, x: NDArray[np.float64]) -> float:
        z = np.dot(x, self.weights) + self.bias
        if self.activation == "sigmoid":
            return round(float(1 / (1 + np.exp(-z))), 5)
        elif self.activation == "relu":
            return round(float(max(0, z)), 5)
        else:
            return -1