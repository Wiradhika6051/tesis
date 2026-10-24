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
        nodes = sample.pruned_cfg.get("nodes", [])
        edges = sample.pruned_cfg.get("edges", [])

        # Original CFG node ID -> row index in node features
        node_to_idx = {
            node.node_id: idx
            for idx, node in enumerate(nodes)
        }

        edge_pairs = [
            (node_to_idx[src], node_to_idx[dst])
            for src, dst in edges
            if src in node_to_idx and dst in node_to_idx
        ]
        # debug
        # Original CFG node IDs, in the order used to create node features
        node_ids = [node.node_id for node in nodes]
        node_id_set = set(node_ids)
        num_nodes = len(nodes)

        # Check whether node IDs match feature-row positions: 0, 1, ..., N-1
        ids_are_contiguous = node_ids == list(range(num_nodes))

        # Check whether every edge references a retained node
        invalid_endpoints = [
            (src, dst)
            for src, dst in edges
            if src not in node_id_set or dst not in node_id_set
        ]

        # Check whether raw edge IDs are valid feature-row indices
        out_of_range_edges = [
            (src, dst)
            for src, dst in edges
            if src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes
        ]

        print("\n=== Pruned CFG Node-ID Diagnostic ===")
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of edges: {len(edges)}")
        print(f"Node IDs contiguous and ordered: {ids_are_contiguous}")
        print(f"First 20 node IDs: {node_ids[:20]}")
        print(f"Edges referencing missing nodes: {invalid_endpoints[:10]}")
        print(f"Edges with out-of-range indices: {out_of_range_edges[:10]}")
        print("=====================================\n")
        # debug end
        if edge_pairs:
            edge_index = torch.tensor(
                edge_pairs,
                dtype=torch.long,
            ).t().contiguous()
        else:
            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
            )
        # if edges:

        #     edge_index = torch.tensor(
        #         edges,
        #         dtype=torch.long
        #     ).t().contiguous()

        # else:

        #     edge_index = torch.empty(
        #         (2, 0),
        #         dtype=torch.long
        #     )

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