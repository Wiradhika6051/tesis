import torch
from torch_geometric.data import Data


class GCNEncoder:

    def __init__(
        self,
        vudenc_vectorizer,
        cfg_vocab
    ):
        self.vudenc_vectorizer = vudenc_vectorizer
        self.cfg_vocab = cfg_vocab

    def encode(self, sample):
        """
        Convert a pruned CFG into a PyG graph.

        Each CFG node contains a variable-length sequence
        of VUDENC Word2Vec vectors.

        Original CFG node IDs are remapped to contiguous
        feature-row indices: 0, 1, ..., num_nodes - 1.
        """

        nodes = sample.pruned_cfg.get("nodes", [])
        edges = sample.pruned_cfg.get("edges", [])

        node_token_vectors = []
        node_types = []

        # Encode every CFG node independently.
        for node in nodes:
            vectors = self.vudenc_vectorizer.text_to_vectors(
                node.text
            )

            node_token_vectors.append(
                torch.tensor(
                    vectors,
                    dtype=torch.float32
                )
            )

            # CFG node type.
            node_type_id = self.cfg_vocab.get(
                node.node_type,
                self.cfg_vocab["<UNK>"]
            )

            node_types.append(node_type_id)

        # Map original CFG node IDs to feature-row indices.
        node_to_idx = {
            node.node_id: idx
            for idx, node in enumerate(nodes)
        }

        # Verify that every edge references a retained node.
        invalid_edges = [
            (src, dst)
            for src, dst in edges
            if src not in node_to_idx or dst not in node_to_idx
        ]

        if invalid_edges:
            raise ValueError(
                "Pruned CFG contains edges referencing "
                f"missing nodes: {invalid_edges[:10]}"
            )

        # Remap edge endpoints to contiguous feature-row indices.
        edge_pairs = [
            (node_to_idx[src], node_to_idx[dst])
            for src, dst in edges
        ]

        if edge_pairs:
            edge_index = torch.tensor(
                edge_pairs,
                dtype=torch.long
            ).t().contiguous()
        else:
            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long
            )

        # Construct PyG graph.
        graph = Data(edge_index=edge_index)

        # Variable-length Word2Vec sequence for each CFG node.
        graph.node_token_vectors = node_token_vectors

        # CFG node types.
        graph.node_types = torch.tensor(
            node_types,
            dtype=torch.long
        )

        # Metadata.
        graph.sample_id = (
            sample.repo,
            sample.parent_commit,
            sample.file_path,
            getattr(sample, "function_start", None),
            getattr(sample, "function_end", None),
            sample.label
        )

        graph.pair_id = (
            sample.repo,
            sample.parent_commit,
            sample.file_path,
            getattr(sample, "function_start", None),
            getattr(sample, "function_end", None)
        )

        graph.repo = sample.repo
        graph.file_path = sample.file_path
        graph.parent_commit = sample.parent_commit
        graph.label = sample.label

        sample.graph = graph

        return sample
