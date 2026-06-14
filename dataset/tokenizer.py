import io
import tokenize

def tokenize_python(code):

    result = []

    try:

        for tok in tokenize.generate_tokens(
            io.StringIO(code).readline
        ):

            result.append(tok.string)

    except Exception:
        pass

    return result