import torch
import torch.nn
from torchtyping import TensorType

class Ops:
    def reshape(self, X: TensorType[float]) -> TensorType[float]:
        # Reshape (M, N) tensor to (M*N/2, 2)
        M, N = X.shape
        return X.reshape(M * N // 2, 2)

    def average(self, X: TensorType[float]) -> TensorType[float]:
        # Compute column-wise mean (average across rows)
        return torch.mean(X, dim=0)

    def concatenate(self, X: TensorType[float], Y: TensorType[float]) -> TensorType[float]:
        # Join two tensors side-by-side along dim=1. (M, N) + (M, M) -> (M, N + M)
        # Use torch.cat((a, b), dim=1)
        return torch.cat((X, Y), dim=1)

    def get_loss(self, X: TensorType[float], Y: TensorType[float]) -> TensorType[float]:
        # MSE
        # Equivalent to torch.nn.functional.mse_loss(prediction, target)
        return ((X - Y) ** 2).mean()


