import numpy as np

# Complete backprop for two layer net
# Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> preds

def backprop(x, W1, b1, W2, b2, y_true):

    # Forward pass
    z1 = W1 @ x + b1
    a1 = np.maximum(z1, 0)
    z2 = W2 @ a1 + b2

    # Loss: MSE = mean((predictions - y_true)^2)
    loss = np.mean((z2 - y_true) ** 2)

    # Backward pass
    db2 = 2 * (z2 - y_true) / len(y_true)
    dW2 = np.outer(db2, a1)
    db1 = (W2.T @ db2) * np.heaviside(z1, 0)  # heaviside is derivative of ReLU and heaviside(ReLU(z_1)) == heaviside(np.maximum(z1, 0)) == heaviside(z1)
    dW1 = np.outer(db1, x)
    
    return {
        'loss': round(loss, 4),
        'dW1': np.round(dW1, 4),
        'db1': np.round(db1, 4),
        'dW2': np.round(dW2, 4),
        'db2': np.round(db2, 4),
    }

# W_1 is (hidden_size, input_size), corresponds to hidden layer with hidden_size neurons
# W_2 is (output_size, hidden_size), corresponds to output layer with output_size neurons
# In general: W is (# neurons in cur layer, # neurons in prev layer)
# Each neuron is a collection weights, one for each neuron of the previous layer, and a bias
# Each neuron in a layer receives the full output of the previous layer 
