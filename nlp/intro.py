import torch
import torch.nn as nn

def get_dataset(positive, negative):
    # build vocab, encode, pad shorter sentences with 0s
    
    sentences = positive + negative
    words = sorted(set(w for s in sentences for w in s.split()))

    vocab = {words[i]:i+1 for i in range(len(words))}

    tensors = [
        torch.tensor([vocab[w] for w in s.split()]) for s in sentences
    ]

    return nn.utils.rnn.pad_sequence(tensors, batch_first=True)