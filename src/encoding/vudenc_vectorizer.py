import numpy as np
import utils.bilstm_myutils as myutils

from gensim.models import Word2Vec


MAX_LENGTH = 200
EMBEDDING_DIM = 200


class VUDENCVectorizer:

    def __init__(
        self,
        w2v_path
    ):

        self.model = Word2Vec.load(
            w2v_path
        )

        self.word_vectors = (
            self.model.wv
        )

        assert (
            self.word_vectors.vector_size
            == EMBEDDING_DIM
        )

    def text_to_vectors(
        self,
        text
    ):
        """
        Convert source code text into
        VUDENC Word2Vec vectors.

        Returns:
            numpy array with shape:
            (num_tokens, 200)

        Returns an empty array when no
        usable Word2Vec tokens exist.
        """

        if text is None:

            return np.empty(
                (
                    0,
                    EMBEDDING_DIM
                ),
                dtype=np.float32
            )

        #
        # Original VUDENC tokenizer.
        #
        tokens = myutils.getTokens(
            str(text)
        )

        vectors = []

        for token in tokens:

            #
            # Original VUDENC behavior.
            #
            if token == " ":
                continue

            #
            # Gensim 3.x vocabulary.
            #
            if token not in self.word_vectors.vocab:
                continue

            vectors.append(
                self.model[token]
            )

        if len(vectors) == 0:

            return np.empty(
                (
                    0,
                    EMBEDDING_DIM
                ),
                dtype=np.float32
            )

        return np.asarray(
            vectors,
            dtype=np.float32
        )

    def sample_to_vectors(
        self,
        sample
    ):
        """
        Convert a pruned Sample into a
        single source-ordered Word2Vec sequence.

        This representation is used by
        the BiLSTM.
        """

        nodes = sample.pruned_cfg.get(
            "nodes",
            []
        )

        #
        # Preserve source-code order.
        #
        nodes = sorted(
            nodes,
            key=lambda node: (
                node.lineno,
                node.node_id
            )
        )

        #
        # Extract code from pruned CFG.
        #
        code_parts = []

        for node in nodes:

            if node.text is None:
                continue

            text = str(
                node.text
            ).strip()

            if text:
                code_parts.append(
                    text
                )

        code = "\n".join(
            code_parts
        )

        return self.text_to_vectors(
            code
        )

    def prepare_dataset(
        self,
        samples
    ):
        """
        Convert samples into padded
        sequences for the BiLSTM.
        """

        X = []
        y = []

        skipped = 0

        for sample in samples:

            if (
                sample.pruned_cfg is None
                or len(
                    sample.pruned_cfg.get(
                        "nodes",
                        []
                    )
                ) == 0
            ):

                print(
                    "[VECTORIZER SKIP] "
                    "empty pruned CFG"
                )

                skipped += 1
                continue

            vectors = self.sample_to_vectors(
                sample
            )

            if len(vectors) == 0:

                print(
                    "[VECTORIZER SKIP] "
                    "no usable Word2Vec tokens"
                )

                skipped += 1
                continue

            #
            # Truncate.
            #
            vectors = vectors[
                :MAX_LENGTH
            ]

            #
            # Zero padding.
            #
            padded = np.zeros(
                (
                    MAX_LENGTH,
                    EMBEDDING_DIM
                ),
                dtype=np.float32
            )

            padded[
                :len(vectors)
            ] = vectors

            X.append(
                padded
            )

            y.append(
                int(sample.label)
            )

        return (
            np.asarray(
                X,
                dtype=np.float32
            ),
            np.asarray(
                y,
                dtype=np.int32
            ),
            skipped
        )