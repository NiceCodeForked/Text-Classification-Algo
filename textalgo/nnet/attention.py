import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, input_dim, embed_dim):
        super(SelfAttention, self).__init__()
        self.qdim, self.kdim, self.vdim = embed_dim, embed_dim, embed_dim
        self.q = nn.Linear(input_dim, self.qdim)
        self.k = nn.Linear(input_dim, self.kdim)
        self.v = nn.Linear(input_dim, self.vdim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_factor = 1 / math.sqrt(self.kdim)

    def forward(self, x, mask=None, return_attention=False):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        
        values, attention = scaled_dot_product(Q, K, V, mask=None)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class MultiheadAttention(nn.Module):
    """
    References
    ----------
    1. https://uvadlc-notebooks.readthedocs.io/en/latest/index.html
    """
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

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

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [batch, head, seq_len, 3*head_dim]
        q, k, v = qkv.chunk(3, dim=-1) # [batch, head, seq_len, head_dim]

        # Determine value outputs
        # values: [batch, head, seq_len, head_dim]
        # attention: [batch, head, seq_len, seq_len]
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [batch, seq_len, head, head_dim]
        values = values.reshape(batch_size, seq_length, self.num_heads*self.head_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention