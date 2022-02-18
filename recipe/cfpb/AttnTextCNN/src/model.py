import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), '..', '..', '..'))

import torch
import torch.nn as nn
from collections import OrderedDict
from textalgo.models import BaseModel, Classifier
from textalgo.nnet import (
    Squeeze, 
    Unsqueeze, 
    SpatialDropout, 
    SelfAttention, 
    MultiheadAttention
)


class AttnTextCNN(BaseModel):
    
    def __init__(
        self, 
        vocab_size, 
        maxlen=64, 
        embed_dim=300, 
        num_heads=10, 
        num_filters=16, 
        kernel_list=[3, 4, 5], 
        dropout=0.1, 
        lin_neurons=128, 
        lin_blocks=2, 
        num_layers=3, 
        layer_norm_eps=1e-5, 
        num_classes=2
    ):
        super(AttnTextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_list = kernel_list
        self.dropout = dropout
        self.lin_neurons = lin_neurons
        self.lin_blocks = lin_blocks
        self.num_layers = num_layers
        self.layer_norm_eps = layer_norm_eps
        self.num_classes = num_classes
        
        self.encoder = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.spatial = SpatialDropout(dropout)
        # self.attention = SelfAttention(embed_dim, embed_dim)
        # self.attention = MultiheadAttention(embed_dim, embed_dim, num_heads)
        self.transformer_encoder = TransformerEncoder(
            embed_dim, embed_dim, num_heads, num_layers, dropout, layer_norm_eps
        )
        self.convs = nn.ModuleList(
            [self.conv_block(maxlen, num_filters, w, embed_dim) for w in kernel_list]
        )
        self.drop = nn.Dropout(dropout)
        self.classifier = Classifier(
            input_size=len(kernel_list)*num_filters, 
            lin_neurons=lin_neurons, 
            lin_blocks=lin_blocks, 
            out_neurons=num_classes
        )
    
    def conv_block(self, maxlen, num_filters, filter_size, embed_dim):
        """
        Shape
        -----
        [batch, maxlen, embed_dim, 1]
        [batch, n_filters, maxlen-filter_size+1, 1]
        [batch, n_filters, maxlen-filter_size+1, 1]
        [batch, n_filters, maxlen-filter_size+1]
        [batch, n_filters, 1]
        [batch, n_filters, 1]
        """
        return nn.Sequential(OrderedDict([
            ('unsqueeze', Unsqueeze(dim=1)), 
            ('conv', nn.Conv2d(1, num_filters, (filter_size, embed_dim))), 
            ('relu', nn.ReLU(inplace=True)), 
            ('squeeze', Squeeze(dim=3)), 
            ('pool', nn.MaxPool1d(maxlen-filter_size+1, stride=1)), 
            ('bn', nn.BatchNorm1d(num_filters))
        ]))
    
    def forward(self, x):
        """
        Input: [batch, maxlen]
        Output: [batch, n_classes]

        Shape
        -----
        [batch, maxlen, embed_dim]
        [batch, len(kernel_list)*n_filters, 1]
        [batch, len(kernel_list)*n_filters]
        [batch, len(kernel_list)*n_filters]
        [batch, 1, len(kernel_list)*n_filters]
        [batch, 1, n_classes]
        [batch, n_classes]
        """
        x = self.encoder(x)
        x = self.spatial(x)
        x = self.transformer_encoder(x)
        x = torch.cat([layer(x) for layer in self.convs], dim=1)
        x = x.squeeze(2)
        x = self.drop(x)
        x = x.unsqueeze(1)
        x = self.classifier(x)
        return x.squeeze(1)

    def get_model_args(self):
        """ Arguments needed to re-instantiate the model."""
        model_args = {
            "vocab_size": self.vocab_size, 
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim, 
            "num_heads": self.num_heads, 
            "num_filters": self.num_filters,
            "kernel_list": self.kernel_list, 
            "dropout": self.dropout, 
            "lin_neurons": self.lin_neurons, 
            "lin_blocks": self.lin_blocks, 
            "num_layers": self.num_layers, 
            "layer_norm_eps": self.layer_norm_eps, 
            "num_classes": self.num_classes
        }
        return model_args


class TransformerEncoder(nn.Module):
    """
    References
    ----------
    1. https://zhuanlan.zhihu.com/p/127030939
    """
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob, eps=1e-12):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob, 
                                                  eps=eps)
                                     for _ in range(n_layers)])

    def forward(self, x, s_mask=None):
        for layer in self.layers:
            x = layer(x, s_mask)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, eps=1e-12):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(d_model, d_model, n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model, eps=eps)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, s_mask):
        _x = x
        x = self.attention(x, mask=s_mask, return_attention=False)
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out