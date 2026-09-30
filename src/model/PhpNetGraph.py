import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    global_max_pool
)


class PhpNetGraph(nn.Module):

    def __init__(
        self,
        cfg_vocab_size,
        word2vec_dim=200
    ):

        super().__init__()

        #
        # =====================================================
        # CFG node type embedding
        # =====================================================
        #
        self.cfg_embedding = nn.Embedding(
            num_embeddings=cfg_vocab_size,
            embedding_dim=32
        )

        #
        # =====================================================
        # Encode each CFG node from its Word2Vec sequence.
        #
        # Input:
        #     200-d VUDENC Word2Vec
        #
        # Output:
        #     64 forward + 64 backward = 128
        # =====================================================
        #
        self.gru = nn.GRU(
            input_size=word2vec_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        #
        # =====================================================
        # Graph Encoder
        #
        # 128-d GRU representation
        # +
        # 32-d CFG type representation
        # =
        # 160-d node representation
        # =====================================================
        #
        self.conv1 = GCNConv(
            160,
            128
        )

        self.conv2 = GCNConv(
            128,
            256
        )

        self.conv3 = GCNConv(
            256,
            256
        )

        #
        # =====================================================
        # Graph classifier
        # =====================================================
        #
        self.classifier = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                128,
                2
            )
        )

    def encode_node(
        self,
        node_vectors
    ):
        """
        Encode one CFG node.

        Input:
            [num_tokens, 200]

        Output:
            [128]
        """

        #
        # No usable Word2Vec tokens.
        #
        if node_vectors.size(0) == 0:

            return torch.zeros(
                128,
                dtype=torch.float32,
                device=node_vectors.device
            )

        #
        # Add batch dimension.
        #
        # [num_tokens, 200]
        #
        # becomes:
        #
        # [1, num_tokens, 200]
        #
        node_vectors = (
            node_vectors.unsqueeze(0)
        )

        #
        # BiGRU.
        #
        _, hidden = self.gru(
            node_vectors
        )

        #
        # hidden shape:
        #
        # [num_layers * directions,
        #  batch,
        #  hidden_size]
        #
        # = [4, 1, 64]
        #
        # Last layer forward:
        #
        forward = hidden[
            -2,
            0
        ]

        #
        # Last layer backward:
        #
        backward = hidden[
            -1,
            0
        ]

        #
        # 64 + 64 = 128
        #
        return torch.cat(
            (
                forward,
                backward
            ),
            dim=0
        )

    def build_node_features(
        self,
        graph
    ):

        #
        # =====================================================
        # Encode every CFG node.
        # =====================================================
        #

        token_features = []

        for node_vectors in (
            graph.node_token_vectors
        ):

            node_vectors = (
                node_vectors.to(
                    graph.node_types.device
                )
            )

            feature = self.encode_node(
                node_vectors
            )

            token_features.append(
                feature
            )

        #
        # [num_nodes, 128]
        #
        token_feature = torch.stack(
            token_features,
            dim=0
        )

        #
        # =====================================================
        # CFG node type embedding
        # =====================================================
        #

        cfg_feature = self.cfg_embedding(
            graph.node_types
        )

        #
        # [num_nodes, 32]
        #
        #
        # 128 + 32 = 160
        #
        x = torch.cat(
            (
                token_feature,
                cfg_feature
            ),
            dim=1
        )

        return x

    def forward(
        self,
        graph
    ):

        #
        # =====================================================
        # Build node representations.
        # =====================================================
        #

        x = self.build_node_features(
            graph
        )

        #
        # =====================================================
        # Graph structure.
        # =====================================================
        #

        edge_index = (
            graph.edge_index
        )

        batch = graph.batch

        #
        # =====================================================
        # GCN
        # =====================================================
        #

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv3(
            x,
            edge_index
        )

        x = F.relu(x)

        #
        # =====================================================
        # Graph embedding
        # =====================================================
        #

        x = global_max_pool(
            x,
            batch
        )

        #
        # =====================================================
        # Classification
        # =====================================================
        #

        logits = self.classifier(
            x
        )

        return logits