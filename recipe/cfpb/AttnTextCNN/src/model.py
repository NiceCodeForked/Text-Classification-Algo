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
        self.num_classes = num_classes
        
        self.encoder = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.spatial = SpatialDropout(dropout)
        self.attention = SelfAttention(embed_dim, embed_dim)
        self.attention = MultiheadAttention(embed_dim, embed_dim, num_heads)
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
        x = self.attention(x)
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
            "num_filters": self.num_filters,
            "kernel_list": self.kernel_list, 
            "dropout": self.dropout, 
            "lin_neurons": self.lin_neurons, 
            "lin_blocks": self.lin_blocks, 
            "num_classes": self.num_classes
        }
        return model_args