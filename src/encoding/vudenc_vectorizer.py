import numpy as np

class VUDENCVectorizer:

    def __init__(self, w2v_path):

        data = np.load(
            w2v_path,
            allow_pickle=True
        )

        self.tokens = data["tokens"]
        self.vectors = data["vectors"]

        self.word_to_index = {
            token: index
            for index, token in enumerate(
                self.tokens
            )
        }

    def get_vector(self, token):

        index = self.word_to_index.get(token)

        if index is None:
            return None

        return self.vectors[index]