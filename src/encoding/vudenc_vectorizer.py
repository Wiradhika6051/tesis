import numpy as np

import utils.bilstm_myutils as myutils


EMBEDDING_DIM = 200


class VUDENCVectorizer:

    def __init__(
        self,
        w2v_path
    ):

        data = np.load(
            w2v_path,
            allow_pickle=True
        )

        self.tokens = data[
            "tokens"
        ]

        self.vectors = data[
            "vectors"
        ]

        assert (
            self.vectors.shape[1]
            == EMBEDDING_DIM
        )

        self.word_to_index = {

            token: index

            for index, token
            in enumerate(
                self.tokens
            )

        }

    def get_vector(
        self,
        token
    ):
        """
        Return the Word2Vec vector for
        a token.

        Returns None for OOV tokens.
        """

        index = self.word_to_index.get(
            token
        )

        if index is None:

            return None

        return self.vectors[
            index
        ]

    def text_to_vectors(
        self,
        text
    ):
        """
        Convert source code text into
        VUDENC Word2Vec vectors.

        This preserves the original VUDENC
        tokenization and OOV behavior.

        Returns:
            numpy array with shape:
            (num_tokens, 200)
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

            vector = self.get_vector(
                token
            )

            #
            # Original VUDENC behavior:
            # skip OOV tokens.
            #
            if vector is None:

                continue

            vectors.append(
                vector
            )

        if not vectors:

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