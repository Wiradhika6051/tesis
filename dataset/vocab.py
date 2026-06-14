from collections import Counter

from tokenizer import tokenize_python


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


def encode_code(
    code,
    vocab
):

    return [
        vocab.get(
            token,
            vocab["<UNK>"]
        )
        for token in tokenize_python(code)
    ]