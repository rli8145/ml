# self-attention

# T = num_tokens/seq_length, d = embedding_dim/model_dim
# each head of attention has W_Q, W_K, W_V. (d, d_k), d_k = attention_dim 
# X is (T, d)
# Q = X @ W_Q, K = X @ W_K, V = X @ W_V. (T, d) @ (d, d_k) -> (T, d_k) i.e. each token is projected into a query/key/value vector
# K, Q are determine attention weights via Q @ K.T and are symmetric in role (could swap and get same weights but transposed)
# V extracts value vector for each token once attention is decided

# Attention:
# Q @ K.T:              (T, d_k) @ (d_k, T) -> (T, T) matrix of dot products!
# softmax(QK.T/√dk) (T, T) - dividing by √dk normalizes variance. axis=1, i.e. softmax each row individually
# output:            softmax(Q @ K.T / √dk) @ V = (T, T) @ (T, d_k) -> (T, d_k)


# notes
# - intuition: embedding diff as a function of context
# - masking to prevent later tokens from influencing earlier ones
# - attention pattern size = context size ^ 2. bottleneck

import torch
import torch.nn as nn
from torchtyping import TensorType

# T = num_tokens/seq_length, d = embedding_dim, d_k = attention_dim
class SingleHeadAttention(nn.Module):

    def __init__(self, d: int, d_k: int):
        super().__init__()
        torch.manual_seed(0)

        self.key = nn.Linear(d, d_k, bias=False)
        self.query = nn.Linear(d, d_k, bias=False)
        self.value = nn.Linear(d, d_k, bias=False)

    def forward(self, X): # X is (T, d)
        K, Q, V = self.key(X), self.query(X), self.value(X)
        # (batch size, T, d_k)

        T, d_k = K.shape[1], K.shape[2]

        scores = (Q @ K.transpose(-2, -1)) / d_k ** 0.5

        mask = torch.tril(torch.ones(T, T))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        return weights @ V # round in multi_head forward