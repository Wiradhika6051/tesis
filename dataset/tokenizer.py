import io
import tokenize
from tesis.dataset.vocab import tokenize_code


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

            tokens.append({
                "token": tok.string,
                "line": tok.start[0]
            })

    except Exception:
        print("Tokenizer error:", e)
        return []

    return tokens




def encode_code(
    code,
    kept_lines,
    vocab
):

    encoded = []

    for token, line in tokenize_code(
        code,
        with_lines=True
    ):

        if line not in kept_lines:
            continue

        encoded.append(
            vocab.get(
                token,
                vocab["<UNK>"]
            )
        )

    return encoded

MAX_LEN = 4096
def pad_sequence(seq):

    if len(seq) > MAX_LEN:

        return seq[:MAX_LEN]

    return seq + (
        [0] * (MAX_LEN - len(seq))
    )
    