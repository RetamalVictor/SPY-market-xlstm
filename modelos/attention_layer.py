import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

def scaled_dot_product(q, k, v):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention



class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim
        self.head_dim = embed_dim 

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, return_attention=True):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, 3*self.head_dim)
        qkv = qkv.permute(0, 1, 2) # [Batch, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v)
        values = values.permute(0, 1, 2) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
        

if __name__ == "__main__":
    
    seq_len, d_k = 3, 2
    pl.seed_everything(42)
    q = torch.randn(seq_len, d_k)
    k = torch.randn(seq_len, d_k)
    v = torch.randn(seq_len, d_k)
    values, attention = scaled_dot_product(q, k, v)
    print("Q\n", q)
    print("K\n", k)
    print("V\n", v)
    print("Values\n", values)
    print("Attention\n", attention)

    # Test multihead attention
    pl.seed_everything(42)
    x = torch.randn(2, 4, 8)
    mha = MultiheadAttention(8, 16)
    o, attention = mha(x)
    print("Input\n", x)
    print("Output\n", o)
    print("Output shape\n", o.shape)
    print("Attention shape\n", attention.shape)

