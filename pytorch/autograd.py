import numpy as np

# tensor class with autograd supporting addition, multiplication, relu

# sample usage:

# forward
# z = x * y + x.relu()
#
# backward — computes gradients (i.e. loss.backward() in pytorch)
# z.backward()
#
# update weights (i.e. optimizer.step() in pytorch)
# x.data -= lr * x.grad
# y.data -= lr * y.grad
#
# zero gradients for next iteration (i.e. optimizer.zero_grad() in pytorch)
# x.grad = 0.0
# y.grad = 0.0


class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=float)
        self.grad = 0.0
        self._backward = lambda: None  # how to compute local gradient
        self._prev = set(_children) # inputs to this node
        self._op = _op # for debugging

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        # ∂L/∂self += ∂out/∂self * ∂L/∂out. ∂out/∂self = 1
        def _backward():
            self.grad += out.grad 
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        # ∂L/∂self += ∂out/∂self * ∂L/∂out. ∂out/∂self = other
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad # heaviside
        out._backward = _backward
        return out

    def backward(self):
        # topological sort
        nodes = []
        visited = set()
        def topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topo(child)
                nodes.append(v)
        topo(self)

        self.grad = 1.0  # dL/dL = 1
        for node in reversed(nodes): # parents before children
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

# += in _backward() needed to handle graphs with shared nodes. if multiple paths to z from x: 
# z = h(f(x), g(x)) -> ∂z/∂x = ∂z/∂f * ∂f/∂x + ∂z/∂g * ∂g/∂x

# Example:
# x = Tensor(3.0)
# z = x * x
# z.backward()
# print(x.grad)

# x → (self)  ↘
#              * → z
# x → (other) ↗

# Vanilla neural network with distinct weights: x → W1 → ReLU → W2 → ReLU → W3 → ReLU → ... → Wn → preds
# each Wi is used exactly once, so each Wi.grad gets += exactly once during backward.
# if some W repeated, then += needed for accumulation