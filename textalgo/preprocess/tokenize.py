import re
import pysbd
import itertools
import warnings


unicode = str
PATTERN = r""" (?x)             # set flag to allow verbose regexps
        (?:[A-Z]\.)+            # abbreviations, e.g. U.S.A.
        | \$?\d+(?:\.\d+)?%?    # currency and percentages, $12.40, 50%
        | \w+(?:-\w+)+          # words with internal hyphens
        | \w+(?:'[a-z])         # words with apostrophes
        | \.\.\.                # ellipsis
        |(?:Mr|Mrs|Dr|Ms)\.     # honorifics
        | \w+                   # normal words
        """
warnings.filterwarnings('ignore')


def tokenize(text, flatten=True, encoding='utf8', errors='strict'):
    """
    Tokenise a text into a sequence of words.

    Parameters
    ----------
    text: str
        Input text may be either unicode or utf8-encoded byte string.
    """
    
    text = any2unicode(text, encoding=encoding, errors=errors)
    # Sentence boundary disambiguation
    seg = pysbd.Segmenter(language="en", clean=False)
    tokens = [re.findall(PATTERN, sentence) for sentence in seg.segment(text)]
    # Tokenise into word-level tokens
    if flatten:
        return list(itertools.chain(*tokens))
    return tokens


def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)