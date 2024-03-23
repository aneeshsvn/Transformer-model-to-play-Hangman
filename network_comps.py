import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange, reduce, repeat


class Positional_Encoding(nn.Module):
    def __init__(self, max_len:int, embedding_dim:int, dropout=0.1):
        super().__init__()
        numer = torch.arange(max_len)[:, None]
        denom = torch.exp(2*torch.arange(0, embedding_dim, 2)*(-math.log(10000.0)/embedding_dim))[None, :]
        p_encoding = torch.empty((1, max_len, embedding_dim))
        p_encoding[0, :, 0::2] = torch.sin(numer*denom)
        p_encoding[0, :, 1::2] = torch.cos(numer*denom)
        self.register_buffer('p_encoding', p_encoding)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        #x should have the shape [batch_size, seq_len, embedding_dim]
        x = x + self.p_encoding
        return self.dropout(x)
    
    
class EncoderLayer(nn.Sequential):
    def __init__(self, embedding_dim, num_heads, d_fc, dropout=0.1):
        super().__init__(
        ResConnect(nn.Sequential( nn.LayerNorm(embedding_dim), MultiHeadAttention(embedding_dim, num_heads, dropout) )
                   ),
        ResConnect(nn.Sequential( nn.LayerNorm(embedding_dim), Position_FeedForward(embedding_dim, d_fc, dropout) )
                   )
                        )

class ClassificationLayer(nn.Module):
    def __init__(self, max_len:int, embedding_dim:int , vocab_size:int, hidden_layer_sizes:list[int]):
        super().__init__()
        layer_sizes = [embedding_dim*max_len] + hidden_layer_sizes + [vocab_size]
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.Dropout(0.1))
            if i < len(layer_sizes)-2:
                layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers)
                 
    def forward(self, x):
        x = rearrange(x, "b n d -> b (n d)")
        x = nn.LogSoftmax(dim=1)(self.layers(x))
        return x
    
    
#taken from https://einops.rocks/pytorch-examples.html
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.keys = nn.Linear(embedding_dim, embedding_dim)
        self.queries = nn.Linear(embedding_dim, embedding_dim)
        self.values = nn.Linear(embedding_dim, embedding_dim)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.embedding_dim ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    
class Position_FeedForward(nn.Module): 
    def __init__(self, embedding_size: int, d_fc: int, dropout:float): 
        super().__init__() 
        self.linear_1 = nn.Linear(embedding_size, d_fc) 
        self.linear_2 = nn.Linear(d_fc, embedding_size) 
        self.gelu = nn.GELU() 
        self.dropout = nn.Dropout(dropout) 
 
    def forward(self, x): 
        x = self.linear_1(x) 
        x = self.gelu(x) 
        x = self.linear_2(x) 
        x = self.dropout(x) 
        return x
    
class ResConnect(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x