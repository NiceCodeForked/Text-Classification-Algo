import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from textalgo.nnet import Squeeze, Unsqueeze
from textalgo.nnet import Sequential
from textalgo.nnet import Linear
from textalgo.nnet import BatchNorm1d as _BatchNorm1d


class TextCNN(nn.Module):
    """
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
        return nn.Sequential(OrderedDict([
            ('unsqueeze', Unsqueeze(dim=1)),                             # [batch, maxlen, emb_dim, 1]
            ('conv', nn.Conv2d(1, num_filters, (filter_size, emb_dim))), # [batch, n_filters, maxlen-filter_size+1, 1]
            ('relu', nn.ReLU(inplace=True)),                             # [batch, n_filters, maxlen-filter_size+1, 1]
            ('squeeze', Squeeze(dim=3)),                                 # [batch, n_filters, maxlen-filter_size+1]
            ('pool', nn.MaxPool1d(maxlen-filter_size+1, stride=1)),      # [batch, n_filters, 1]
            ('bn', nn.BatchNorm1d(num_filters))                          # [batch, n_filters, 1]
        ]))
    
    def forward(self, x):
        """
        Input: [batch, maxlen]
        Output: [batch, len(kernel_list)*num_filters]
        """
        x = self.encoder(x)                                      # [batch, maxlen, emb_dim]
        x = torch.cat([layer(x) for layer in self.convs], dim=1) # [batch, len(kernel_list)*n_filters, 1]
        x = x.squeeze(2)                                         # [batch, len(kernel_list)*n_filters]
        x = self.drop(x)                                         # [batch, len(kernel_list)*n_filters]
        x = x.unsqueeze(1)                                       # [batch, 1, len(kernel_list)*n_filters]
        x = self.classifier(x)                                   # [batch, 1, n_classes]
        return x.squeeze(1)


class Classifier(torch.nn.Module):
    """This class implements the cosine similarity on the top of features.
    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of classes.
    Example
    -------
    >>> classify = Classifier(input_size=2, lin_neurons=2, out_neurons=2)
    >>> outputs = torch.tensor([ [1., -1.], [-9., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outupts = outputs.unsqueeze(1)
    >>> cos = classify(outputs)
    >>> (cos < -1.0).long().sum()
    tensor(0)
    >>> (cos > 1.0).long().sum()
    tensor(0)
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
        """Returns the output probabilities over speakers.
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