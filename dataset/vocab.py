from collections import Counter

from tesis.dataset.tokenizer import tokenize_python


def build_vocab(
    samples,
    min_freq=2
):

    counter = Counter()

    for sample in samples:

        counter.update(
            tokenize_python(
                sample["code"]
            )
        )

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for token, freq in counter.items():

        if freq >= min_freq:
            vocab[token] = len(vocab)

    return vocab

MAX_LEN = 4096
def encode_code(
    code,
    vocab
):

    tokens = tokenize_python(code)

    ids = [
        vocab.get(
            token,
            vocab["<UNK>"]
        )
        for token in tokens
    ]

    if len(ids) > MAX_LEN:

        ids = ids[:MAX_LEN]

    else:

        ids += [0] * (
            MAX_LEN - len(ids)
        )

    return ids