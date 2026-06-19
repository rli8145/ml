# self-attention

# T = num_tokens/seq_length, d = embedding_dim/model_dim
# each head of attention has W_Q, W_K, W_V. (d, d_k), d_k = attention_dim 
# X is (T, d)
# Q = X @ W_Q, K = X @ W_K, V = X @ W_V. (T, d) @ (d, d_k) -> (T, d_k) i.e. each token is mapped to a query/key/value vector

# Attention:
# Q @ K.T:  (T, d_k) @ (d_k, T) -> (T, T) matrix of dot products!
# weights: softmax(QK.T/√dk) (T, T) - dividing by √dk normalizes variance. axis=1, i.e. softmax each row 
# output:  weights @ V = (T, T) @ (T, d_k) -> (T, d_k). for each of the d_k dimensions, take the corresponding values from each token (T,) and take T weighted sums with the softmaxed dot products

# Q, K technically symmetric in role (could swap and get same weights but transposed). row i, col j = how much token i attends to token j in the context of the this head
# V extracts value vector for each token

# notes
# - intuitive example with adjectives and "big red dog"
#       "dog" queries via Q, "big" and "red" advertise themselves as relevant adjectives via K
#       -> high Q·K relevance scores for (dog, big) and (dog, red). softmax gives high attention weights to "big" and "red"
#       -> adjust dog's embedding with weighted sum of all V vectors, heavily weighted toward V_big and V_red
#       -> dog's new embedding incorporates "big-ness" and "red-ness"
# - masking to prevent later tokens from influencing earlier ones
# - attention pattern ((T, T) weight matrix) size = T^2. bottleneck which motivates sparse attention, flash attention, etc
# - backprop trains everything from W_O, W_V, W_Q, W_K to embedding table (vocab_size, d). error = cross-entropy loss

import torch
import torch.nn as nn

# T = num_tokens/seq_length, d = embedding_dim, d_k = attention_dim
class SingleHeadAttention(nn.Module):
    def __init__(self, d, d_k):
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
