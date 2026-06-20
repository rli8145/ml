import numpy as np

# tensor class with autograd supporting addition, multiplication, relu

# sample usage:

# forward
# z = x * y + x.relu()
#
# backward — computes gradients (loss.backward() in pytorch)
# z.backward()
#
# update weights (optimizer.step() in pytorch)
# x.data -= lr * x.grad
# y.data -= lr * y.grad
#
# zero gradients for next iter (optimizer.zero_grad() in pytorch)
# x.grad = 0.0
# y.grad = 0.0


class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=float)
        self.grad = 0.0
        self._backward = lambda: None  # how to compute local gradient
        self._prev = set(_children) # inputs to this node
        self._op = _op

    # z = h(f(x), g(x)): ∂z/∂x = ∂z/∂f * ∂f/∂x + ∂z/∂g * ∂g/∂x
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad # ∂/∂x (x+y) = 1
            other.grad += out.grad # ∂/∂y (x+y) = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad # ∂/∂x (x*y) = y
            other.grad += self.data * out.grad # ∂/∂y (x*y) = x
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
        for node in reversed(nodes):
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"