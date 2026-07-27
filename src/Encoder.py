from tesis.preprocessing.vocabulary import tokenize_code


class Encoder:

    def __init__(
        self,
        token_vocab,
        cfg_vocab
    ):
        self.token_vocab = token_vocab
        self.cfg_vocab = cfg_vocab

    def encode(
        self,
        sample
    ):

        token_ids = []
        node_type_ids = []

        for node in sample.pruned_cfg["nodes"]:

            #
            # Encode source-code tokens.
            #
            tokens = tokenize_code(node.text)

            encoded_tokens = [

                self.token_vocab.get(
                    token,
                    self.token_vocab["<UNK>"]
                )

                for token in tokens
            ]

            token_ids.append(
                encoded_tokens
            )

            #
            # Encode CFG node type.
            #
            node_type_ids.append(

                self.cfg_vocab.get(
                    node.node_type,
                    self.cfg_vocab["<UNK>"]
                )

            )

        sample.tokens = token_ids
        sample.node_types = node_type_ids
        sample.edges = sample.pruned_cfg["edges"]

        return sample