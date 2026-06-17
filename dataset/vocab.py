from collections import Counter
import ast
import re


SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>"
]


def tokenize_code(
    code
):

    return re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]",
        code
    )


def build_token_vocab(
    samples,
    min_freq=1
):
    """
    Build vocab for source code tokens.
    Used by GRU branch.
    """

    counter = Counter()

    for sample in samples:

        source = sample.get(
            "source",
            sample.get(
                "code",
                ""
            )
        )

        counter.update(
            tokenize_code(
                source
            )
        )

    vocab = {
        token: idx
        for idx, token in enumerate(
            SPECIAL_TOKENS
        )
    }

    for token, count in counter.items():

        if count >= min_freq:

            if token not in vocab:

                vocab[token] = len(
                    vocab
                )

    return vocab

def build_cfg_vocab():
    """
    Build vocabulary for CFG nodes.
    Used by GCN branch.
    """

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    cfg_node_types = [

        "Assign",
        "AugAssign",

        "Call",

        "If",
        "For",
        "While",

        "Return",

        "Try",
        "ExceptHandler",

        "With",

        "FunctionDef",
        "AsyncFunctionDef",

        "ClassDef",

        "Expr",

        "Break",
        "Continue",

        "Raise",

        "Import",
        "ImportFrom",

        "Pass"
    ]

    for node_type in cfg_node_types:

        vocab[node_type] = len(
            vocab
        )

    return vocab