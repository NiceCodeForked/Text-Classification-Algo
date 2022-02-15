import itertools
from collections import Counter
from textalgo.preprocess.tokenize import tokenize


# This is still under development
def build_vocab_from_txt_file(
    text_file, 
    vocab_size=100, 
    do_lower_case=False, 
    min_frequency=None, 
    max_frequency=None, 
    added_tokens=[{'id':0, 'content':'[PAD]'}, {'id':1, 'content':'[UNK]'}]
):
    # Build the counter from txt file
    tokens = []
    with open(text_file, 'r') as f:
        for line in f.readlines():
            if line.replace('\n', ''):
                # Tokenise the sentence into tokens and count
                tokens.append(tokenize(line))
    tokens = list(itertools.chain(*tokens))
    counter = Counter(tokens)

    # Handle min_frequency and max_frequency
    if not min_frequency:
        min_frequency = 0
    if not max_frequency:
        max_frequency = counter.most_common(1)[0][1]
    counter = {k:v for k,v in counter.items() if (v >= min_frequency) and (v <= max_frequency)}
    print(f'Total number of vocabulary is {len(counter)}')

    # Sort the counter by frequency in descending order
    counter = sorted(counter, key=counter.get, reverse=True)[:(vocab_size-len(added_tokens))]
    vocab = [word.lower() if do_lower_case else word for word in sorted(counter)]

    # Combine special tokens and original vocab
    special_token_max_id = max([s['id'] for s in added_tokens])
    assert special_token_max_id < vocab_size, 'Special token ID must smaller than vocab size!'

    vocab_tmp = [None] * vocab_size
    for d in added_tokens:
        vocab_tmp[d['id']] = d['content']
    positions = [i for i in range(len(vocab_tmp)) if vocab_tmp[i] is None]
    for pos, tkn in zip(positions, vocab):
        vocab_tmp[pos] = tkn
    token2index = {v:k for k, v in enumerate(vocab_tmp) if v is not None}
    index2vocab = {k:v for k, v in enumerate(vocab_tmp) if v is not None}
    assert max(token2index.values()) <= vocab_size, (
        f'Special token ID must smaller than vocab size! '
        f'Please adjust min_frequency and max_frequency to proper values.'
    )
    return token2index, index2vocab