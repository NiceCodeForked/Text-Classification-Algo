import re
import pysbd
import string
import itertools
import warnings
import unicodedata
import concurrent.futures as cf
from pathlib import Path
from typing import List, Union, Iterable
from typeguard import check_argument_types


ALL_LETTERS = string.ascii_letters + " .,;'-"
PATTERN = r"""
    (?x)                    # set flag to allow verbose regexps
    (?:[A-Z]\.)+            # abbreviations, e.g. U.S.A.
    | \$?\d+(?:\.\d+)?%?    # currency and percentages, $12.40, 50%
    | \w+(?:-\w+)+          # words with internal hyphens
    | \w+(?:'[a-z])         # words with apostrophes
    | \.\.\.                # ellipsis
    |(?:Mr|mr|Mrs|mrs|Dr|dr|Ms|ms)\.     # honorifics
    | \w+                   # normal words
    | [,.!?\\-]             # specific punctuation
"""


def word_tokenize(text):

    def tokenise_(t):
        warnings.filterwarnings('ignore')
        t = any2unicode(t)
        # Sentence boundary disambiguation
        seg = pysbd.Segmenter(language="en", clean=False)
        tokens = [re.findall(PATTERN, sentence) for sentence in seg.segment(t)]
        return list(itertools.chain(*tokens))

    if isinstance(text, str):
        return tokenise_(text)
    elif isinstance(text, list):
        # with cf.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        #     result = executor.map(tokenise_, text)
        # return result
        return [tokenise_(t) for t in text]


def char_tokenize():
    pass


def build_tokenizer(
    token_type, 
    special_tokens: Union[str, Iterable[str]] = None, 
    remove_special_tokens: bool = False, 
    non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
    remove_non_linguistic_symbols: bool = False,
    space_symbol: str = "<space>", 
    uncased: bool = False, 
    delimiter: str = None
):
    if token_type == "char":
        return CharTokenizer(
            uncased=uncased, 
            non_linguistic_symbols=non_linguistic_symbols,
            space_symbol=space_symbol,
            remove_non_linguistic_symbols=remove_non_linguistic_symbols,
        )
    elif token_type == "word":
        return WordTokenizer(
            uncased=uncased, 
            delimiter=delimiter, 
            special_tokens=special_tokens, 
            remove_special_tokens=remove_special_tokens
        )
    else:
        raise ValueError(
            f"token_type must be either char or word: " f"{token_type}"
        )


class CharTokenizer(object):
    """
    Tokenise a text into a sequence of characters.

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
            f'uncased={self.normalise}, '
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


class WordTokenizer(object):
    """
    Tokenise a text into a sequence of words.

    Parameters
    ----------
    uncased: bool
        Whether the text being lowercased.
    delimiter: str
        A delimiter is a sequence of one or more characters for 
        specifying the boundary between separate.
    special_tokens: List[str]
        A list of special tokens. Every special token should be 
        started with a left square bracket and ended with right 
        square bracket. e.g. "[PAD]"
    remove_special_tokens: bool
        Whether to remove special tokens in the sentence.
    """
    def __init__(
        self, 
        uncased: bool = False, 
        delimiter: str = None,
        special_tokens: Union[str, Iterable[str]] = None,
        remove_special_tokens: bool = False,
    ):
        assert check_argument_types()
        self.normalise = uncased
        self.delimiter = delimiter

        if special_tokens is None:
            self.special_tokens = set()
        else:
            self.special_tokens = set(special_tokens)
        self.remove_special_tokens = remove_special_tokens

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'uncased="{self.uncased}", '
            f'delimiter="{self.delimiter}", '
            f'remove_special_tokens={self.remove_special_tokens}'
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
        for t in self.split(line, self.special_tokens):
            if self.remove_special_tokens and t in self.special_tokens:
                continue
            # Lowercase the token if normalise is True (except for special tokens)
            t = t.lower() if (self.normalise) and (t not in self.special_tokens) else t
            tokens.append(t)
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        if self.delimiter is None:
            delimiter = " "
        else:
            delimiter = self.delimiter
        return delimiter.join(tokens)

    @staticmethod
    def split(
        text, 
        special_tokens=None, 
        pattern=PATTERN, 
        flatten=True, 
        encoding='utf8', 
        errors='strict'
    ):
        warnings.filterwarnings('ignore')
        text = any2unicode(text, encoding=encoding, errors=errors)
        # Sentence boundary disambiguation
        seg = pysbd.Segmenter(language="en", clean=False)
        # Add special tokens into regex pattern
        if special_tokens:
            exclude = set(string.punctuation)
            special_tokens = [
                r'\['+''.join(ch for ch in s if ch not in exclude)+r'\]' 
                for s in special_tokens
            ]
            results = '| '.join(special_tokens)
            pattern = f"{pattern}|(?:{results})"
        # Start tokenising the sentence into word-level tokens
        tokens = [re.findall(pattern, sentence) for sentence in seg.segment(text)]
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
    unicode = str
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
