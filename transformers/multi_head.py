# multi-head:
# X projected down to d_k = d/num_heads. intuitively, each head operates on a smaller subspace of embedding dimension and attends to a different aspect of input - syntax, semantics, ..., these are emergent. 
# heads run in parallel, produce num_heads x (T, d_k) then concacenated to (T, d). W_O (d, d) then mixes across heads: (T, d) @ (d, d) -> (T, d)
# since d_k = d/num_heads, total computation is same as single-headed attention with full dimension

import torch
import torch.nn as nn
from single_head import SingleHeadAttention

# T = num_tokens/seq_length, d = embedding_dim, d_k = attention_dim
class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, d, d_k, num_heads):
        super().__init__()
        torch.manual_seed(0)
        self.d_k = d_k // num_heads
        self.heads = nn.ModuleList([SingleHeadAttention(d, self.d_k) for _ in range(num_heads)])  
        self.proj = nn.Linear(d_k, d_k, bias=False) # W_O


    def forward(self, X):
        out = torch.cat([head(X) for head in self.heads], dim=2)
        # (batch, T, attention_dim) @ (attention_dim, attention_dim) -> (batch, T, attention_dim)
        return torch.round(self.proj(out), decimals=4)
