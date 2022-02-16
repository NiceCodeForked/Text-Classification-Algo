import math
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from abc import abstractmethod
from textalgo.vectors import Vectors


class EmbeddingAggregator(nn.Module):
    """
    Such convolution will have time_steps parameters. 
    Each parameter will be a equal to embed_dim.
    In other words this convolution will ran over dimension 
    with size embed_dim and sum it with learnable weights.

    Parameters
    ----------
    in_channels: int
        This is the size of the time steps.

    References
    ----------
    1. https://stackoverflow.com/a/58574603
    """
    def __init__(self, in_channels):
        self.aggregator = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=1, 
            kernel_size=1
        )

    def forward(self, x):
        """
        x: [batch_size, time_steps, embed_dim]
        """
        return self.aggregator(x)


class BaseEmbedding(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError


class GloveGensimEmbedding(BaseEmbedding):

    def __init__(
        self, 
        pretrained_model_name, 
        word2idx, 
        init_type='random', 
        norm=True, 
        padding_idx=None, 
        trainable=True
    ):
        super(GloveGensimEmbedding, self).__init__()
        self.pretrained_model_name = pretrained_model_name
        self.word2idx = word2idx
        self.init_type = init_type
        self.norm = norm
        self.padding_idx = padding_idx

        self.vocab_size = len(word2idx)
        self.weights = make_embeddings_matrix(
            pretrained_model_name, word2idx, init_type, norm
        )
        self.dim = self.weights.shape[1]
        self.embedding = nn.Embedding(self.vocab_size, self.dim, padding_idx=padding_idx)
        self.embedding.weight.data = torch.Tensor(self.weights)
        if not trainable:
            self.embedding.weight.requires_grad = False

    def forward(self, x):
        embeds = self.embedding(x)
        return embeds


def make_embeddings_matrix(
    pretrained_model_name, 
    word2idx, 
    init_type='random', 
    norm=True
):
    """
    Create embeddings matrix to use in Embedding layer.

    Parameters
    ----------
    pretrained_model_name: str
        A pretrained model hosted inside a model repo on huggingface hub.
        Check it on https://huggingface.co/models?sort=downloads&search=fse%2F
        Aliases
            - glove-twitter-25
            - glove-twitter-50
            - glove-twitter-100
            - glove-twitter-200
            - glove-wiki-gigaword-50
            - glove-wiki-gigaword-100
            - glove-wiki-gigaword-200
            - glove-wiki-gigaword-300
            - paragram-25
            - paragram-300-sl999
            - paragram-300-ws353
            - paranmt-300
            - word2vec-google-news-300
    word2idx: Dict
        A dictionary that maps word to index.
    init_type: str
        The way to initialise the embedding matrix.
    norm: bool
        If True, the resulting vector will be L2-normalized (unit Euclidean length).

    References
    ----------
    1. https://madewithml.com/courses/foundations/embeddings/
    """
    # Load from gensim pre-trained model
    print(f'Loading {pretrained_model_name}...')
    wv = Vectors.from_pretrained(pretrained_model_name)
    embedding_dim = wv.vector_size
    gensim_word2idx = wv.key_to_index
    gensim_word2vec = {
        k: wv.get_vector(k, norm=norm) for k, v in tqdm(gensim_word2idx.items())
    }
    del wv

    # Generate embedding matrix for nn.Embedding layer
    print(f'Building embedding matrix...')
    if init_type == 'random':
        embedding_matrix = np.random.randn(len(word2idx), embedding_dim)
        embedding_matrix = np.divide(
            embedding_matrix, 
            np.linalg.norm(embedding_matrix, axis=1, ord=2)[:, np.newaxis]
        )
    else:
        embedding_matrix = np.zeros((len(word2idx), embedding_dim))
    
    for word, i in word2idx.items():
        embedding_vector = gensim_word2vec.get(word, None)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


class PositionalEmbedding(nn.Module):
    """
    Learned Positional Embedding

    Examples
    --------
    >>> vocab_size, embed_dim = 10, 300
    >>> batch_size, max_length = 4, 16
    >>> tok_embed = nn.Embedding(vocab_size, embed_dim)
    >>> pos_embed = PositionEmbedding(max_length, embed_dim)
    >>> norm = nn.LayerNorm(embed_dim)
    >>> x = torch.randint(10, (batch_size, max_length))
    >>> y = norm(tok_embed(x) + pos_embed(x))
    >>> print(y.shape)
    torch.Size([4, 16, 300])

    References
    ----------
    1. https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial
    2. https://aclanthology.org/2021.emnlp-main.236.pdf
    """
    def __init__(self, max_length, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.pos_embed = nn.Embedding(max_length, embed_dim)

    def forward(self, x):
        """
        Inputs: [batch_size, max_length]
        Outputs: [batch_size, max_length, embed_dim]
        """
        seq_length = x.size(1)
        pos = torch.arange(seq_length, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)
        return self.pos_embed(pos)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding which uses sine and cosine functions.

    Parameters
    ----------
    embed_dim: int
        The embedding dimension (required).
    dropout: float
        The dropout value (default=0.1).
    max_len: int
        The maximum length of the incoming sequence (default=5000).
    
    Examples
    --------
    >>> batch_size, max_length, embed_dim = 4, 16, 300
    >>> pos_encoder = PositionalEncoding(embed_dim)
    >>> x = torch.randint(10, (batch_size, max_length, embed_dim))
    >>> y = pos_embed(x)
    >>> print(y.shape)
    torch.Size([4, 16, 300])

    References
    ----------
    1. https://github.com/pytorch/pytorch/issues/51551
    """
    def __init__(self, embed_dim, p=0.1, max_length=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=p)
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Inputs: [batch_size, seq_length, embed_dim]
        Outputs: [batch_size, seq_length, embed_dim]
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x.permute(1, 0, 2)