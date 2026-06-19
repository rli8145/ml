import numpy as np

def forward(x, gamma, beta):
    # x: 1D feature vector
    # gamma: 1D scale parameter (same length as x)
    # beta: 1D shift parameter (same length as x)
    # eps = 1e-5
    # Normalize: x_hat = (x - mean) / sqrt(var + eps)
    # Scale and shift: out = gamma * x_hat + beta
    
    eps = 1e-5
    x_hat = (x - np.mean(x)) / np.sqrt(np.var(x) + eps)

    return np.round(gamma * x_hat + beta, 5)