import torch

from torch_geometric.data import Data

from src.vocabulary import tokenize_code


class Encoder:

    def __init__(
        self,
        token_vocab,
        cfg_vocab,
        max_tokens_per_node=128
    ):

        self.token_vocab = token_vocab
        self.cfg_vocab = cfg_vocab

        self.max_tokens_per_node = (
            max_tokens_per_node
        )

    def encode(
        self,
        sample
    ):

        node_tokens = []
        node_types = []

        #
        # --------------------------------------------------
        # Encode CFG nodes
        # --------------------------------------------------
        #

        for node in sample.pruned_cfg["nodes"]:

            tokens = tokenize_code(
                node.text
            )

            encoded = [

                self.token_vocab.get(
                    token,
                    self.token_vocab["<UNK>"]
                )

                for token in tokens

            ]

            #
            # Truncate very large nodes.
            #
            encoded = encoded[
                :self.max_tokens_per_node
            ]

            #
            # Pad node to fixed length.
            #
            encoded += [

                self.token_vocab["<PAD>"]

            ] * (

                self.max_tokens_per_node
                -
                len(encoded)

            )

            node_tokens.append(
                encoded
            )

            node_types.append(

                self.cfg_vocab.get(

                    node.node_type,

                    self.cfg_vocab["<UNK>"]

                )

            )

        #
        # --------------------------------------------------
        # Build edge_index
        # --------------------------------------------------
        #

        edges = sample.pruned_cfg["edges"]

        if edges:

            edge_index = torch.tensor(
                edges,
                dtype=torch.long
            ).t().contiguous()

        else:

            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long
            )

        #
        # --------------------------------------------------
        # Build PyG graph
        # --------------------------------------------------
        #

        graph = Data(

            edge_index=edge_index

        )

        #
        # Node token IDs
        #
        graph.node_tokens = torch.tensor(

            node_tokens,

            dtype=torch.long

        )

        #
        # CFG node type IDs
        #
        graph.node_types = torch.tensor(

            node_types,

            dtype=torch.long

        )

        #
        # Store graph
        #
        sample.graph = graph

        return sample