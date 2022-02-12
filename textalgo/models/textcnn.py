import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from textalgo.nnet import Squeeze, Unsqueeze
from textalgo.nnet import Linear
from textalgo.nnet import BatchNorm1d as _BatchNorm1d
from textalgo.nnet import SeparableConv2D


class TextCNN(nn.Module):
    """
    Yoon Kim, Convolutional neural networks for sentence classification (2014)

    References
    ----------
    1. https://arxiv.org/pdf/1408.5882.pdf
    """
    def __init__(
        self, 
        vocab_size, 
        maxlen, 
        emb_dim, 
        num_filters=16, 
        kernel_list=[3, 4, 5], 
        dropout=0.1, 
        lin_neurons=128, 
        lin_blocks=2, 
        num_classes=2
    ):
        super(TextCNN, self).__init__()
        self.maxlen = maxlen
        self.emb_dim = emb_dim
        self.num_filters = num_filters
        self.kernel_list = kernel_list
        
        self.encoder = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [self.conv_block(maxlen, num_filters, w, emb_dim) for w in kernel_list]
        )
        self.drop = nn.Dropout(dropout)
        self.classifier = Classifier(
            input_size=len(kernel_list)*num_filters, 
            lin_neurons=lin_neurons, 
            lin_blocks=lin_blocks, 
            out_neurons=num_classes
        )
    
    def conv_block(self, maxlen, num_filters, filter_size, emb_dim):
        """
        [batch, maxlen, emb_dim, 1]
        [batch, n_filters, maxlen-filter_size+1, 1]
        [batch, n_filters, maxlen-filter_size+1, 1]
        [batch, n_filters, maxlen-filter_size+1]
        [batch, n_filters, 1]
        [batch, n_filters, 1]
        """
        return nn.Sequential(OrderedDict([
            ('unsqueeze', Unsqueeze(dim=1)), 
            ('conv', nn.Conv2d(1, num_filters, (filter_size, emb_dim))), 
            ('relu', nn.ReLU(inplace=True)), 
            ('squeeze', Squeeze(dim=3)), 
            ('pool', nn.MaxPool1d(maxlen-filter_size+1, stride=1)), 
            ('bn', nn.BatchNorm1d(num_filters))
        ]))
    
    def forward(self, x):
        """
        Input: [batch, maxlen]
        Output: [batch, len(kernel_list)*num_filters]

        Shape
        -----
        [batch, maxlen, emb_dim]
        [batch, len(kernel_list)*n_filters, 1]
        [batch, len(kernel_list)*n_filters]
        [batch, len(kernel_list)*n_filters]
        [batch, 1, len(kernel_list)*n_filters]
        [batch, 1, n_classes]
        [batch, n_classes]
        """
        x = self.encoder(x)
        x = torch.cat([layer(x) for layer in self.convs], dim=1)
        x = x.squeeze(2)
        x = self.drop(x)
        x = x.unsqueeze(1)
        x = self.classifier(x)
        return x.squeeze(1)


class LightWeightedTextCNN(TextCNN):
    """
    Ritu Yadav, Light-Weighted CNN for Text Classification (2020)

    References
    ----------
    1. https://arxiv.org/pdf/2004.07922.pdf
    """
    def conv_block(self, maxlen, num_filters, filter_size, emb_dim):
        """
        [batch, maxlen, emb_dim, 1]
        [batch, n_filters, maxlen-filter_size+1, 1]
        [batch, n_filters, maxlen-filter_size+1, 1]
        [batch, n_filters, maxlen-filter_size+1]
        [batch, n_filters, 1]
        [batch, n_filters, 1]
        """
        return nn.Sequential(OrderedDict([
            ('unsqueeze', Unsqueeze(dim=1)), 
            ('conv', SeparableConv2D(1, num_filters, (filter_size, emb_dim))), 
            ('relu', nn.ReLU(inplace=True)), 
            ('squeeze', Squeeze(dim=3)), 
            ('pool', nn.MaxPool1d(maxlen-filter_size+1, stride=1)), 
            ('bn', nn.BatchNorm1d(num_filters)) 
        ]))
    

class Classifier(nn.Module):
    """
    Parameters
    ----------
    device : str
        Device used, e.g., "cpu" or "cuda".
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of classes.
    """
    def __init__(
        self,
        input_size,
        device="cpu",
        lin_blocks=0,
        lin_neurons=192,
        out_neurons=1211,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        for block_index in range(lin_blocks):
            self.blocks.extend(
                [
                    _BatchNorm1d(input_size=input_size),
                    Linear(input_size=input_size, n_neurons=lin_neurons),
                ]
            )
            input_size = lin_neurons

        # Final Layer
        self.weight = nn.Parameter(
            torch.FloatTensor(out_neurons, input_size, device=device)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor
            Torch tensor.
        """
        for layer in self.blocks:
            x = layer(x)

        # Need to be normalized
        x = F.linear(F.normalize(x.squeeze(1)), F.normalize(self.weight))
        return x.unsqueeze(1)