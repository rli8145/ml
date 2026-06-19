import numpy as np

def softmax(z):
    e = np.exp(z - z.max())
    return np.round(e / e.sum(), 4)

# Temp = 1