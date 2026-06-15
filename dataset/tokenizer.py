import io
import tokenize

def tokenize_python(code):

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

def encode_code(
    code,
    vocab
):

    tokens = tokenize_python(code)

    return [
        vocab.get(
            token,
            vocab["<UNK>"]
        )
        for token in tokens
    ]

MAX_LEN = 4096
def pad_sequence(seq):

    if len(seq) > MAX_LEN:

        return seq[:MAX_LEN]

    return seq + (
        [0] * (MAX_LEN - len(seq))
    )