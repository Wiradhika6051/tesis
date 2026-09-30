import torch

from torch_geometric.data import Data


class GCNEncoder:

    def __init__(
        self,
        vudenc_vectorizer,
        cfg_vocab
    ):

        self.vudenc_vectorizer = (
            vudenc_vectorizer
        )

        self.cfg_vocab = cfg_vocab

    def encode(
        self,
        sample
    ):
        """
        Convert a pruned CFG into a PyG graph.

        Each CFG node contains a variable-length
        sequence of VUDENC Word2Vec vectors.

        node_token_vectors:
            [
                Tensor(num_tokens_node_0, 200),
                Tensor(num_tokens_node_1, 200),
                ...
            ]
        """

        nodes = sample.pruned_cfg.get(
            "nodes",
            []
        )

        edges = sample.pruned_cfg.get(
            "edges",
            []
        )

        node_token_vectors = []
        node_types = []

        #
        # Encode every CFG node independently.
        #
        for node in nodes:

            vectors = (
                self.vudenc_vectorizer
                .text_to_vectors(
                    node.text
                )
            )

            node_token_vectors.append(
                torch.tensor(
                    vectors,
                    dtype=torch.float32
                )
            )

            #
            # CFG node type.
            #
            node_type_id = (
                self.cfg_vocab.get(
                    node.node_type,
                    self.cfg_vocab["<UNK>"]
                )
            )

            node_types.append(
                node_type_id
            )

        #
        # Encode CFG edges.
        #
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
        # Construct PyG graph.
        #
        graph = Data(
            edge_index=edge_index
        )

        #
        # Variable-length Word2Vec sequence
        # for every CFG node.
        #
        graph.node_token_vectors = (
            node_token_vectors
        )

        #
        # CFG node types.
        #
        graph.node_types = torch.tensor(
            node_types,
            dtype=torch.long
        )

        #
        # Metadata.
        #
        graph.sample_id = (
            sample.repo,
            sample.parent_commit,
            sample.file_path,
            getattr(
                sample,
                "function_start",
                None
            ),
            getattr(
                sample,
                "function_end",
                None
            ),
            sample.label
        )

        graph.pair_id = (
            sample.repo,
            sample.parent_commit,
            sample.file_path,
            getattr(
                sample,
                "function_start",
                None
            ),
            getattr(
                sample,
                "function_end",
                None
            )
        )

        graph.repo = sample.repo
        graph.file_path = sample.file_path
        graph.parent_commit = (
            sample.parent_commit
        )
        graph.label = sample.label

        sample.graph = graph

        return sample