import numpy as np
from linear_regression import LinearRegression

X = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10]
], dtype=np.float64)

Y = np.array([8, 18, 28, 38, 48], dtype=np.float64) 

model = LinearRegression(learning_rate=0.001, iters=10000)
model.fit(X, Y)
print("Learned weights:", model.weights) # should be close to [2, 3]
print("Prediction:", model.prediction(X))
print("Error:", model.error(model.prediction(X), Y))