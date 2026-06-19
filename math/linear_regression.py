import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, iters=1000):
        self.learning_rate = learning_rate
        self.iters = iters
        self.weights = None

    def prediction(self, X):
        # X is (n, m), weights is (m,) -> return (n,) predictions (n = # samples, m = # features)
        # absorb bias by appending column 
        return np.round(X @ self.weights, 5)
        
    def error(self, mp, target):
        # diagnostic
        mse = ((mp - target) ** 2).sum() / len(mp)
        return round(mse, 5)
    
    def derivative(self, mp, target, X, j):
        # partial derivative of mse with respect to w_j
        return -2 * np.dot(target - mp, X[:, j]) / len(X)
        # note len(X) == len(mp) == len(Y)

    def fit(self, X, Y, initial_weights=None):
        if initial_weights is None:
            self.weights = np.zeros(X.shape[1]) # X.shape[1] = no. of features
        else:
            self.weights = initial_weights
        
        for _ in range(self.iters):
            mp = self.prediction(X)
            gradients = np.array([self.derivative(mp, Y, X, j) for j in range(len(self.weights))])
            self.weights -= self.learning_rate * gradients

# in context of neural nets, this is a network with no hidden layers and one output neuron with m features (=> m weights)