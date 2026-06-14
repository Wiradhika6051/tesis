import io
import pickle
import tokenize

from collections import Counter


class PythonTokenizer:

    def tokenize(self, code):

        result = []

        try:

            for tok in tokenize.generate_tokens(
                io.StringIO(code).readline
            ):

                tok_type = tokenize.tok_name[
                    tok.type
                ]

                if tok_type in (
                    "NL",
                    "NEWLINE",
                    "INDENT",
                    "DEDENT",
                    "ENDMARKER"
                ):
                    continue

                result.append(tok.string)

        except Exception:
            pass

        return result

    def build_vocab(
        self,
        samples,
        min_freq=2
    ):

        counter = Counter()

        for sample in samples:

            tokens = self.tokenize(
                sample["code"]
            )

            counter.update(tokens)

        vocab = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        for token, freq in counter.items():

            if freq >= min_freq:

                vocab[token] = len(vocab)

        return vocab

    def save_vocab(
        self,
        vocab,
        path
    ):

        with open(path, "wb") as f:
            pickle.dump(vocab, f)

    def load_vocab(
        self,
        path
    ):

        with open(path, "rb") as f:
            return pickle.load(f)

    def encode(
        self,
        code,
        vocab
    ):

        tokens = self.tokenize(code)

        return [
            vocab.get(
                token,
                vocab["<UNK>"]
            )
            for token in tokens
        ]