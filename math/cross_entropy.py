import numpy as np
from numpy.typing import NDArray

eps = 1e-7

def binary_cross_entropy(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    loss = -(y_true * np.log(y_pred + eps) + (1-y_true) * np.log(1-y_pred + eps)).sum()/len(y_true)
    return round(loss, 4)

def categorical_cross_entropy(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    loss = -np.sum(y_true * np.log(y_pred + eps)) / y_true.shape[0]
    return round(loss, 4)