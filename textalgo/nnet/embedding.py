import torch
import torch.nn as nn
from abc import abstractmethod
from ._embedding import GloVe


class BaseEmbedding(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError


class GloveEmbedding(BaseEmbedding):
    """ 
    Wrapper class for text generating RNN 
    Aliases
    -------
    'glove.42B.300d': dim='300', name='42B'
    'glove.6B.100d': dim='100', name='6B'
    'glove.6B.200d': dim='200', name='6B'
    'glove.6B.300d': dim='300', name='6B'
    'glove.6B.50d': dim='50', name='6B'
    'glove.840B.300d': dim='300', name='840B'
    'glove.twitter.27B.100d': dim='100', name='twitter.27B'
    'glove.twitter.27B.200d': dim='200', name='twitter.27B'
    'glove.twitter.27B.25d': dim='25', name='twitter.27B'
    'glove.twitter.27B.50d': dim='50', name='twitter.27B'
    """
    def __init__(self, vocab=None, name="840B", dim=300, trainable=False, init_type='zero'):
        super(GloveEmbedding, self).__init__()

        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.name = name
        self.dim = dim
        if init_type == 'zero':
            vectors = GloVe(name=self.name, dim=self.dim, unk_init=torch.Tensor.zero_)
        elif init_type == 'uniform':
            vectors = GloVe(name=self.name, dim=self.dim, unk_init=torch.Tensor.uniform_)

        self.weights = torch.zeros(self.vocab_size, vectors.dim)

        for i, idx in enumerate(list(self.vocab.idx2word.keys())):
            self.weights[i, :] = vectors[self.vocab[idx]]

        self.embedding = nn.Embedding(self.vocab_size, self.dim, padding_idx=self.vocab.pad_token_id)
        self.embedding.weight.data = torch.Tensor(self.weights)

        if not trainable:
            self.embedding.weight.requires_grad = False

    def forward(self, batch):
        embeds = self.embedding(batch)
        return embeds


class SimpleEmbedding(BaseEmbedding):
    """ Wrapper class for text generating RNN """
    def __init__(self, vocab=None, dim=300, trainable=False):
        super(SimpleEmbedding, self).__init__()

        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.dim = dim

        self.embedding = nn.Embedding(self.vocab_size, self.dim)

        if not trainable:
            self.embedding.weight.requires_grad = False

    def forward(self, batch):
        embeds = self.embedding(batch)
        return embeds