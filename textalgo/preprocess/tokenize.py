import re
import pysbd
import string
import itertools
import warnings
import unicodedata
from pathlib import Path
from typing import List, Union, Iterable
from typeguard import check_argument_types


unicode = str
ALL_LETTERS = string.ascii_letters + " .,;'-"
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


def build_tokenizer(token_type):
    pass


class CharTokenizer(object):
    """
    Parameters
    ----------
    uncased: bool
        Whether the text being lowercased.
    space_symbol: str
        Space symbol default is '<space>'
    non_linguistic_symbols: Path, str, List[str]
        Symbol to be removed.
    remove_non_linguistic_symbols: bool
        Whether to remove non linguistic symbols.

    References
    ----------
    1. https://github.com/espnet/espnet
    """
    def __init__(
        self,
        uncased: bool = True, 
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
    ):
        assert check_argument_types()
        if not remove_non_linguistic_symbols and non_linguistic_symbols is not None:
            raise AttributeError(
                "non_linguistic_symbols is only used "
                "when remove_non_linguistic_symbols = True"
            )
        self.normalise = uncased
        self.space_symbol = space_symbol
        if non_linguistic_symbols is None:
            self.non_linguistic_symbols = set()
        elif isinstance(non_linguistic_symbols, (Path, str)):
            non_linguistic_symbols = Path(non_linguistic_symbols)
            with non_linguistic_symbols.open("r", encoding="utf-8") as f:
                self.non_linguistic_symbols = set(line.rstrip() for line in f)
        else:
            self.non_linguistic_symbols = set(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'normalise={self.normalise}, '
            f'space_symbol="{self.space_symbol}", '
            f'non_linguistic_symbols="{self.non_linguistic_symbols}", '
            f'remove_non_linguistic_symbols={self.remove_non_linguistic_symbols}'
            f")"
        )

    def __call__(self, input_, *args, **kwds):
        if isinstance(input_, str):
            return self.text2tokens(input_, *args, **kwds)
        elif all(map(lambda x: isinstance(x, str), input_)):
            return self.tokens2text(input_, *args, **kwds)
        else:
            raise TypeError('Input should be either List[str] or str!')

    def text2tokens(self, line: str) -> List[str]:
        tokens = []
        while len(line) != 0:
            for w in self.non_linguistic_symbols:
                if line.startswith(w):
                    if not self.remove_non_linguistic_symbols:
                        tokens.append(line[: len(w)])
                    line = line[len(w) :]
                    break
            else:
                t = line[0]
                if t == " ":
                    t = "<space>"
                t = t.lower() if self.normalise else t
                tokens.append(t)
                line = line[1:]
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        tokens = [t if t != self.space_symbol else " " for t in tokens]
        return "".join(tokens)


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
    """
    Convert a string (bytestring in `encoding` or unicode), to unicode.
    
    References
    ----------
    1. https://tedboy.github.io/nlps/_modules/gensim/utils.html#any2unicode
    """
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


def unicode2ascii(text):
    """
    Turn a Unicode string to plain ASCII

    References
    ----------
    1. https://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )
