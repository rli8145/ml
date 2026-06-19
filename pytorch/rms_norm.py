import numpy as np

def rms_norm(x, gamma, eps):
    # RMS Normalization

    x, gamma = np.array(x), np.array(gamma)
    rms = np.sqrt(1 / x.size * np.sum(x ** 2) + eps)
    x /= rms

    return list(np.round(x * gamma, 4))
