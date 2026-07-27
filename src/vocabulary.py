from collections import Counter
import io
import tokenize


SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>"
]


def iter_cfg_nodes(samples):
    """
    Iterate over all nodes in all pruned CFGs.
    """

    for sample in samples:

        if sample.pruned_cfg is None:
            continue

        yield from sample.pruned_cfg["nodes"]


def tokenize_code(code):
    """
    Tokenize source code into lexical tokens.
    """

    tokens = []

    try:

        for tok in tokenize.generate_tokens(
            io.StringIO(code).readline
        ):

            tok_type = tokenize.tok_name[tok.type]

            if tok_type in (
                "NL",
                "NEWLINE",
                "INDENT",
                "DEDENT",
                "ENDMARKER"
            ):
                continue

            tokens.append(tok.string)

    except Exception:
        pass

    return tokens


def build_token_vocab(
    samples,
    min_freq=1
):
    """
    Build vocabulary for source-code tokens.
    """

    counter = Counter()

    for node in iter_cfg_nodes(samples):

        counter.update(
            tokenize_code(node.text)
        )

    vocab = {
        token: idx
        for idx, token in enumerate(
            SPECIAL_TOKENS
        )
    }

    for token, count in counter.items():

        if count >= min_freq:

            vocab.setdefault(
                token,
                len(vocab)
            )

    return vocab


def build_cfg_vocab(samples):
    """
    Build vocabulary for CFG node types.
    """

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for node in iter_cfg_nodes(samples):

        vocab.setdefault(
            node.node_type,
            len(vocab)
        )

    return vocab