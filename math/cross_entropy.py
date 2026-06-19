import numpy as np

eps = 1e-7

def binary_cross_entropy(y_true, y_pred):
    loss = -(y_true * np.log(y_pred + eps) + (1-y_true) * np.log(1-y_pred + eps)).sum()/len(y_true)
    return round(loss, 4)

def categorical_cross_entropy(y_true, y_pred):
    loss = -np.sum(y_true * np.log(y_pred + eps)) / y_true.shape[0]
    return round(loss, 4)