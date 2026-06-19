import numpy as np

def train(X, y, epochs, lr):
    # X: (n_samples, n_features)
    # y: (n_samples,) targets
    # epochs: number of training iterations
    # lr: learning rate
    #
    # Model: y_hat = X @ w + b
    # Loss: MSE = (1/n) * sum((y_hat - y)^2)
    # Initialize w = zeros, b = 0
    # return (np.round(w, 5), round(b, 5))
    n_samples, n_features = X.shape
    w, b = np.zeros(n_features), 0
    for _ in range(epochs):
        y_hat = X @ w + b
        dw = 2 / n_samples * X.T @ (y_hat - y)
        db = 2 / n_samples * np.sum(y_hat - y)
        w -= (dw * lr)
        b -= (db * lr)

    mse = 1 / n_samples * np.sum((y_hat - y) ** 2)
    return np.round(w, 5), round(b, 5), round(mse, 5)