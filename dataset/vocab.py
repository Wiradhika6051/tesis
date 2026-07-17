from collections import Counter
import io
import tokenize


SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>"
]


def tokenize_code(
    code,
    with_lines=False
):
    """
    Tokenize Python source.

    Parameters
    ----------
    with_lines : bool
        False -> ["if", "x", "=", ...]
        True  -> [("if", 1), ("x", 1), ("=", 1), ...]
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

            if with_lines:

                tokens.append(
                    (
                        tok.string,
                        tok.start[0]
                    )
                )

            else:

                tokens.append(
                    tok.string
                )

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

    for sample in samples:

        source = sample.get(
            "source",
            sample.get(
                "code",
                ""
            )
        )

        counter.update(
            tokenize_code(source)
        )

    vocab = {
        token: idx
        for idx, token in enumerate(
            SPECIAL_TOKENS
        )
    }

    for token, count in counter.items():

        if (
            count >= min_freq
            and token not in vocab
        ):

            vocab[token] = len(vocab)

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