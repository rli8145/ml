import numpy as np

# backward pass for a single neuron with sigmoid activation and MSE loss
def backward(x, w, b, y_true):
    # x: 1D input array
    # w: 1D weight array

    # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
    # Loss: L = 0.5 * (y_hat - y_true)^2
    z = np.dot(w, x) + b
    y_hat = 1/(1 + np.exp(-z)) # sigmoid
        
    dL_dw = (y_hat - y_true) * y_hat * (1 - y_hat) * x
    dL_db = (y_hat - y_true) * y_hat * (1 - y_hat) 
    return (np.round(dL_dw, 5), round(dL_db, 5))

    #∂L/∂w_i = ∂L/∂y_hat x ∂y_hat/∂z x ∂z/∂w_i = (y_hat - y) x (e^{-z}/(1+e^{-z}) x 1/(1+e^{-z})) x x_i 
    #                                          = (y_hat - y) y_hat(1-y_hat) x_i 
    #∂z/∂b = 1