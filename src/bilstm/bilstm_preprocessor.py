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


    def sample_to_vectors(
        self,
        sample
    ):
        """
        Convert a pruned Sample into a sequence
        of VUDENC Word2Vec vectors.
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
        # Extract code from the pruned CFG.
        #
        code_parts = []

        for node in nodes:

            text = node.text

            if text is None:
                continue

            text = str(text).strip()

            if text:
                code_parts.append(
                    text
                )

        code = "\n".join(
            code_parts
        )

        #
        # Use the original VUDENC tokenizer.
        #
        tokens = myutils.getTokens(
            code
        )

        vectors = []

        for token in tokens:

            if token == " ":
                continue

            #
            # Gensim 3.x vocabulary.
            #
            if token not in self.word_vectors.vocab:
                continue

            vector = self.model[
                token
            ]

            vectors.append(
                vector
            )

        if len(vectors) == 0:
            return None

        return np.asarray(
            vectors,
            dtype=np.float32
        )


    def prepare_dataset(
        self,
        samples
    ):
        """
        Convert a list of processed Samples
        into padded Word2Vec sequences.
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
                print("[VECTORIZER SKIP] empty pruned CFG")
                skipped += 1
                continue

            vectors = self.sample_to_vectors(
                sample
            )
            if vectors is None:
                print("[VECTORIZER SKIP] no usable Word2Vec tokens")
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