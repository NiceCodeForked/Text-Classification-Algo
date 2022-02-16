import math
import torch
import operator
import itertools
from collections import Counter
from torch.utils.data import Dataset


class TextTfidfDataset(Dataset):
    """
    Parameters
    ----------
    texts: List[List[str]] 
        A list of list of tokenised words.
    labels: List[int]
        A list of labels.
    max_length: int
        Maximum length of the sentence.
    ds: str
        Type of the dataset, either 'train' or 'test'
    word2idx: Dict[str]=int
        Word index dictionary that maps word to index.
        Only needed when ds is set to 'test'
    idf: Dict[str]=float
        Inverse Dense Frequency (IDF) score.
        Only needed when ds is set to 'test'

    Examples
    --------
    >>> texts_train = [
    ...     ['it', 'is', 'going', 'to', 'rain', 'today'], 
    ...     ['today', 'I', 'am', 'not', 'going', 'outside'], 
    ...     ['I', 'am', 'going', 'to', 'see', 'the', 'season', 'premiere']
    ... ]
    >>> texts_test = [
    ...     ['it', 'is', 'not', 'gonna', 'happen', 'ever'], 
    ...     ['today', 'I', 'am', 'not', 'going', 'home'], 
    ...     ['I', 'am', 'going', 'back', 'home', 'to', 'play', 'outside']
    ... ]
    >>> labels_train = [0, 1, 2]
    >>> labels_test = [2, 1, 0]
    >>> train_ds = TextTfidfDataset(
    ...     texts_train, labels_train, max_length=8, ds='train'
    ... )
    >>> test_ds = TextTfidfDataset(
    ...     texts_test, 
    ...     labels_test, 
    ...     max_length=8, 
    ...     ds='test', 
    ...     word2idx=train_ds.word2idx, 
    ...     idf=train_ds.idf
    ... )
    >>> train_dl = DataLoader(train_ds, batch_size=4)
    >>> test_dl = DataLoader(test_ds, batch_size=4)
    >>> pprint(next(iter(train_dl)))
    >>> pprint(next(iter(test_dl)))

    References
    ----------
    1. https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    """
    def __init__(
        self, 
        texts, 
        labels, 
        max_length, 
        ds='train', 
        word2idx=None, 
        idf=None
    ):
        self.ds = ds
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.num_documents = len(self.texts)
        if ds == 'train':
            self._train_prepare()
            self._fit_tfidf()
        elif ds == 'test':
            self._test_prepare(word2idx, idf)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        input_ids = [self.word2idx.get(token, 1) for token in text]
        attention_mask = [1] * len(input_ids)
        # Compute Term Frequency (TF)
        counter = Counter(text)
        denominator = sum(counter.values())
        counter = {k: v/denominator for k, v in counter.items()}
        tf = [counter[token] for token in text]
        # Compute Inverse Dense Frequency (IDF)
        idf = [self.idf.get(token, self.max_idf) for token in text]
        # Padding to max length
        input_ids = self.trimmer(input_ids, self.max_length, 0)
        attention_mask = self.trimmer(attention_mask, self.max_length, 0)
        tf = self.trimmer(tf, self.max_length, 0)
        idf = self.trimmer(idf, self.max_length, 0)
        tfidf = list(map(operator.mul, tf, idf))
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long), 
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long), 
            'tf': torch.tensor(tf), 
            'idf': torch.tensor(idf), 
            'tfidf': torch.tensor(tfidf), 
            'label': torch.tensor(label, dtype=torch.long)
        }

    def _test_prepare(self, word2idx, idf):
        self.vocab = word2idx.keys()
        self.word2idx = word2idx
        self.idx2word = {i:word for word, i in word2idx.items()}
        self._idf = idf
        self.max_idf = max(idf.values())

    def _train_prepare(self):
        self.vocab = set(list(itertools.chain(*self.texts)))
        self.word2idx = {word:i+2 for i, word in enumerate(self.vocab)}
        self.word2idx['[PAD]'] = 0
        self.word2idx['[UNK]'] = 1
        self.idx2word = {i:word for word, i in self.word2idx.items()}

    def _fit_tfidf(self):
        self._idf = {word:0 for word in self.vocab}
        for idx, text in enumerate(self.texts):
            for token in text:
                self._idf[token] += 1
        # Inverse document frequency smooth
        self._idf = {k:math.log10(self.num_documents/(v+1))+1 for k, v in self._idf.items()}
        self.max_idf = max(self._idf.values())

    @property
    def idf(self):
        return self._idf

    @staticmethod
    def trimmer(seq, size, filler=0):
        return seq[:size] + [filler]*(size-len(seq))
