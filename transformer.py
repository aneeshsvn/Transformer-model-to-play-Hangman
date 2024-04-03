import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange, reduce, repeat

from network_comps import Positional_Encoding, EncoderLayer, ClassificationLayer


class Model(nn.Module):
    def __init__(self, vocab_size = 26, max_len=22, embedding_dim=64, d_fc=512):
        '''
        args:
            max_len: int, the maximum number of token in the input sequence
            embedding_dim: int, the dimension of the token embedding
            d_k: int, dimension of the latent space of the attention head
            d_fc: int, dimension of the latent space of the fully connected layer'''
        super().__init__()
        vocab_size = 26 #padding + mask + 26 alphabets
        num_encoders = 16; num_heads = 4
        self.vocab_size = 26 
        self.max_len = max_len; self.embedding_dim = embedding_dim 
        self.num_heads = num_heads; self.f_fc = d_fc #encoder hyperparams
        self.num_encoders = num_encoders
        self.embedding_layer = nn.Embedding(vocab_size+4, embedding_dim, padding_idx=0, max_norm = 1) #embedding layer
        self.positional_encoding = Positional_Encoding(max_len, embedding_dim, dropout=0.1)  #positional encoding
        self.encoders = [EncoderLayer(embedding_dim, num_heads, d_fc) for _ in range(num_encoders)]
        hidden_layer_sizes = [512]
        self.classification_layer = ClassificationLayer(max_len, embedding_dim , vocab_size, hidden_layer_sizes)
        self.network_layers = nn.Sequential(self.embedding_layer, self.positional_encoding, *self.encoders, self.classification_layer)
        
    
    def forward(self, x: Tensor) -> Tensor:
        inputs = x
        x = self.network_layers(x)
        return x
        





