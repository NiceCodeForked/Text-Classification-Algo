import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from abc import abstractmethod
from ._embedding import GloVe
from textalgo.vectors import Vectors


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