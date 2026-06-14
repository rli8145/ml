import numpy as np
from numpy.typing import NDArray

def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
    # embeddings: (vocab_size, embed_dim) matrix
    # token_ids: 1D array of integer token IDs
    # Return the embedding vectors for the given token IDs

    res = []
    for id in token_ids:
        res.append(embeddings[id])
    return np.round(np.array(res), 5)