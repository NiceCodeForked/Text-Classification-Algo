import os
import torch
import torch.nn as nn
from abc import abstractmethod
from .word_vectors import _PretrainedWordVectors


class BaseEmbedding(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError


class GloVe(_PretrainedWordVectors):
    """
    Parameters
    ----------
    name: str 
        Name of the GloVe vectors ('840B', 'twitter.27B', '6B', '42B')
    cache: str (optional)
        Directory for cached vectors
    unk_init: callback (optional)
        By default, initialize out-of-vocabulary word vectors
        to zero vectors; can be any function that takes in a Tensor and
        returns a Tensor of the same size
    is_include: callable (optional)
        Callable returns True if to include a token in memory
        vectors cache; some of these embedding files are gigantic so filtering it can cut
        down on the memory usage. We do not cache on disk if ``is_include`` is defined.
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

    url = {
        '42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        '840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        '6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
    }

    def __init__(self, name='840B', dim=300, **kwargs):
        url = self.url[name]
        name = 'glove.{}.{}d.txt'.format(name, str(dim))
        super(GloVe, self).__init__(name, url=url, **kwargs)


class FastText(_PretrainedWordVectors):
    """ 
    Parameters
    ----------
    language: str
        Language of the vectors
    aligned: bool
        If True: use multilingual embeddings where words with
        the same meaning share (approximately) the same position in the
        vector space across languages. if False: use regular FastText
        embeddings. All available languages can be found under
        https://github.com/facebookresearch/MUSE#multilingual-word-embeddings
    cache: str (optional)
        Directory for cached vectors
    unk_init: callback (optional)
        By default, initialize out-of-vocabulary word vectors
        to zero vectors; can be any function that takes in a Tensor and
        returns a Tensor of the same size
    is_include: callable (optional)
        Callable returns True if to include a token in memory
        vectors cache; some of these embedding files are gigantic so filtering it can cut
        down on the memory usage. We do not cache on disk if ``is_include`` is defined.
    Aliases
    -------
    'fasttext.en.300d': language='en'
    'fasttext.simple.300d': language='simple'
    """
    url_base = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec'
    aligned_url_base = 'https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.{}.align.vec'

    def __init__(self, language="en", aligned=False, **kwargs):
        if aligned:
            url = self.aligned_url_base.format(language)
        else:
            url = self.url_base.format(language)
        name = os.path.basename(url)
        super(FastText, self).__init__(name, url=url, **kwargs)


# List of all 275 supported languages from http://cosyne.h-its.org/bpemb/data/
SUPPORTED_LANGUAGES = [
    'ab', 'ace', 'ady', 'af', 'ak', 'als', 'am', 'an', 'ang', 'ar', 'arc', 'arz', 'as', 'ast',
    'atj', 'av', 'ay', 'az', 'azb', 'ba', 'bar', 'bcl', 'be', 'bg', 'bi', 'bjn', 'bm', 'bn', 'bo',
    'bpy', 'br', 'bs', 'bug', 'bxr', 'ca', 'cdo', 'ce', 'ceb', 'ch', 'chr', 'chy', 'ckb', 'co',
    'cr', 'crh', 'cs', 'csb', 'cu', 'cv', 'cy', 'da', 'de', 'din', 'diq', 'dsb', 'dty', 'dv', 'dz',
    'ee', 'el', 'en', 'eo', 'es', 'et', 'eu', 'ext', 'fa', 'ff', 'fi', 'fj', 'fo', 'fr', 'frp',
    'frr', 'fur', 'fy', 'ga', 'gag', 'gan', 'gd', 'gl', 'glk', 'gn', 'gom', 'got', 'gu', 'gv', 'ha',
    'hak', 'haw', 'he', 'hi', 'hif', 'hr', 'hsb', 'ht', 'hu', 'hy', 'ia', 'id', 'ie', 'ig', 'ik',
    'ilo', 'io', 'is', 'it', 'iu', 'ja', 'jam', 'jbo', 'jv', 'ka', 'kaa', 'kab', 'kbd', 'kbp', 'kg',
    'ki', 'kk', 'kl', 'km', 'kn', 'ko', 'koi', 'krc', 'ks', 'ksh', 'ku', 'kv', 'kw', 'ky', 'la',
    'lad', 'lb', 'lbe', 'lez', 'lg', 'li', 'lij', 'lmo', 'ln', 'lo', 'lrc', 'lt', 'ltg', 'lv',
    'mai', 'mdf', 'mg', 'mh', 'mhr', 'mi', 'min', 'mk', 'ml', 'mn', 'mr', 'mrj', 'ms', 'mt', 'mwl',
    'my', 'myv', 'mzn', 'na', 'nap', 'nds', 'ne', 'new', 'ng', 'nl', 'nn', 'no', 'nov', 'nrm',
    'nso', 'nv', 'ny', 'oc', 'olo', 'om', 'or', 'os', 'pa', 'pag', 'pam', 'pap', 'pcd', 'pdc',
    'pfl', 'pi', 'pih', 'pl', 'pms', 'pnb', 'pnt', 'ps', 'pt', 'qu', 'rm', 'rmy', 'rn', 'ro', 'ru',
    'rue', 'rw', 'sa', 'sah', 'sc', 'scn', 'sco', 'sd', 'se', 'sg', 'sh', 'si', 'sk', 'sl', 'sm',
    'sn', 'so', 'sq', 'sr', 'srn', 'ss', 'st', 'stq', 'su', 'sv', 'sw', 'szl', 'ta', 'tcy', 'te',
    'tet', 'tg', 'th', 'ti', 'tk', 'tl', 'tn', 'to', 'tpi', 'tr', 'ts', 'tt', 'tum', 'tw', 'ty',
    'tyv', 'udm', 'ug', 'uk', 'ur', 'uz', 've', 'vec', 'vep', 'vi', 'vls', 'vo', 'wa', 'war', 'wo',
    'wuu', 'xal', 'xh', 'xmf', 'yi', 'yo', 'za', 'zea', 'zh', 'zu'
]

# All supported vector dimensionalities for which embeddings were trained
SUPPORTED_DIMS = [25, 50, 100, 200, 300]

# All supported number of merge operations for which embeddings were trained
SUPPORTED_MERGE_OPS = [1000, 3000, 5000, 10000, 25000, 50000, 100000, 200000]


class BPEmb(_PretrainedWordVectors):
    """
    Parameters
    ----------
    language: str (optional)
        Language of the corpus on which the embeddings have been trained
    dim: int (optional)
        Dimensionality of the embeddings
    merge_ops: int (optional)
        Number of merge operations used by the tokenizer
    Examples
    --------
    >>> vectors = BPEmb(dim=25)
    >>> subwords = "▁mel ford shire".split()
    >>> vectors[subwords]
    """
    url_base = 'http://cosyne.h-its.org/bpemb/data/{language}/'
    file_name = '{language}.wiki.bpe.op{merge_ops}.d{dim}.w2v.txt'
    zip_extension = '.tar.gz'

    def __init__(self, language='en', dim=300, merge_ops=50000, **kwargs):
        # Check if all parameters are valid
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(("Language '%s' not supported. Use one of the "
                              "following options instead:\n%s") % (language, SUPPORTED_LANGUAGES))
        if dim not in SUPPORTED_DIMS:
            raise ValueError(
                ("Embedding dimensionality of '%d' not supported. "
                 "Use one of the following options instead:\n%s") % (dim, SUPPORTED_DIMS))
        if merge_ops not in SUPPORTED_MERGE_OPS:
            raise ValueError(("Number of '%d' merge operations not supported. "
                              "Use one of the following options instead:\n%s") %
                             (merge_ops, SUPPORTED_MERGE_OPS))

        format_map = {'language': language, 'merge_ops': merge_ops, 'dim': dim}

        # Assemble file name to locally store embeddings under
        name = self.file_name.format_map(format_map)
        # Assemble URL to download the embeddings form
        url = (
            self.url_base.format_map(format_map) + self.file_name.format_map(format_map) +
            self.zip_extension)

        super(BPEmb, self).__init__(name, url=url, **kwargs)