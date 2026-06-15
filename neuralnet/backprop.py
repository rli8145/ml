import numpy as np
from numpy.typing import NDArray
from typing import List

# arbitrary depth

def backprop(self,
             x: NDArray[np.float64],
             weights: List[NDArray[np.float64]],
             biases: List[NDArray[np.float64]],
             y_true: NDArray[np.float64]) -> dict:
    
    # assert len(weights) == len(biases)

    # Forward pass - store intermediates
    zs = []
    activations = [x]
    for W, b in zip(weights, biases):
        z = W @ activations[-1] + b
        zs.append(z)
        a = np.maximum(z, 0) if W is not weights[-1] else z  # no ReLU on last layer
        activations.append(a)

    # Loss
    loss = np.mean((activations[-1] - y_true) ** 2)

    # Backward pass
    dW_list = []
    db_list = []
    delta = 2 * (activations[-1] - y_true) / len(y_true)  # dL/dz at output

    for i in reversed(range(len(weights))):
        dW_list.append(np.outer(delta, activations[i]))
        db_list.append(delta)
        if i > 0:
            delta = (weights[i].T @ delta) * np.heaviside(zs[i-1], 0)

    return {
        'loss': round(loss, 4),
        'dW': [np.round(dW, 4) for dW in reversed(dW_list)],
        'db': [np.round(db, 4) for db in reversed(db_list)],
    }