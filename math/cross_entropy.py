import numpy as np
from numpy.typing import NDArray

class Solution:
    eps = 1e-7

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        loss = -(y_true * np.log(y_pred + self.eps) + (1-y_true) * np.log(1-y_pred + self.eps)).sum()/len(y_true)
        return round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        loss = -np.sum(y_true * np.log(y_pred + self.eps)) / y_true.shape[0]
        return round(loss, 4)