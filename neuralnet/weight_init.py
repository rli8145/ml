import torch
import math
from typing import List

class Initializer:
    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        torch.manual_seed(0) # same W produced every call
        std = math.sqrt(2/(fan_in + fan_out))
        W = torch.randn(fan_out, fan_in) * std
        # torch.randn(y,x) generates a matrix of shape (y,x) from N(0,1)
        return W.round(decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        torch.manual_seed(0)
        std = math.sqrt(2/fan_in)
        W = torch.randn(fan_out, fan_in) * std
        return W.round(decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        torch.manual_seed(0)
        x = torch.randn(1000, input_dim) # batch of 1000 vectors of size input_dim i.e. (batch_size, features)
        stds = []

        for i in range(num_layers):
            fan_in = input_dim if i == 0 else hidden_dim
            fan_out = hidden_dim

            std = 1
            if init_type == "xavier":
                std = math.sqrt(2/(fan_in + fan_out))
            elif init_type == "kaiming":
                std = math.sqrt(2/fan_in) 
            
            W = torch.randn(fan_out, fan_in) * std
            x = torch.relu(x @ W.T) # (1000, fan_in) x (fan_in, fan_out) -> (1000, fan_out)
            # identity: (AB).T = (B.T)(A.T)
            stds.append(round(x.std().item(), 2))

        return stds