import numpy as np
from numpy.typing import NDArray

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, iters: int = 1000):
        self.learning_rate = learning_rate
        self.iters = iters
        self.weights = None

    def prediction(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # absorb bias by appending column 
        return np.round(X @ self.weights, 5)
        
    def error(self, mp: NDArray[np.float64], target: NDArray[np.float64]) -> float:
        # diagnostic
        mse = ((mp - target) ** 2).sum() / len(mp)
        return round(mse, 5)
    
    def derivative(self, mp: NDArray[np.float64], target: NDArray[np.float64], X: NDArray[np.float64], j: int) -> float:
        # partial derivative of mse with respect to w_j
        return -2 * np.dot(target - mp, X[:, j]) / len(X)

    def fit(self, X: NDArray[np.float64], Y: NDArray[np.float64], initial_weights=None) -> None:
        if initial_weights is None:
            self.weights = np.zeros(X.shape[1]) # X.shape[1] = no. of features
        else:
            self.weights = initial_weights
        
        for _ in range(self.iters):
            mp = self.prediction(X)
            gradients = np.array([self.derivative(mp, Y, X, j) for j in range(len(self.weights))])
            self.weights -= self.learning_rate * gradients