import numpy as np

def lookup(embeddings, token_ids):
    # embeddings: (vocab_size, embed_dim) matrix
    # token_ids: 1D array of integer token IDs
    # Return the embedding vectors for the given token IDs

    res = []
    for id in token_ids:
        res.append(embeddings[id])
    return np.round(np.array(res), 5)