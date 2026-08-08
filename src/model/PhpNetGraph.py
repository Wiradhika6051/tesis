import torch
import torch.nn as nn
import torch.nn.functional as F

from asyncio import graph
from torch_geometric.nn import (
    GCNConv,
    global_max_pool
)


class PhpNetGraph(nn.Module):

    def __init__(
        self,
        token_vocab_size,
        cfg_vocab_size
    ):

        super().__init__()

        #
        # Token embedding
        #
        self.token_embedding = nn.Embedding(
            num_embeddings=token_vocab_size,
            embedding_dim=100
        )

        #
        # CFG node type embedding
        #
        self.cfg_embedding = nn.Embedding(
            num_embeddings=cfg_vocab_size,
            embedding_dim=32
        )

        #
        # Encode each CFG node from its tokens.
        #
        self.gru = nn.GRU(
            input_size=100,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        #
        # Graph Encoder
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
        # Graph classifier
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

    def build_node_features(
        self,
        graph
    ):

        #
        # graph.node_tokens
        #
        # Shape:
        # (num_nodes,
        #  max_tokens_per_node)
        #

        token_x = self.token_embedding(
            graph.node_tokens
        )

        _, hidden = self.gru(
            token_x
        )

        #
        # BiGRU output
        #
        token_feature = torch.cat(
            (
                hidden[-2],
                hidden[-1]
            ),
            dim=1
        )

        #
        # graph.node_types
        #
        cfg_feature = self.cfg_embedding(
            graph.node_types
        )

        #
        # Final node feature
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
        # Build node representations
        #
        print(graph.node_tokens.shape)

        print(graph.node_types.shape)

        print(graph.edge_index.shape)
        x = self.build_node_features(
            graph
        )

        #
        # Graph structure
        #
        edge_index = graph.edge_index
        batch = graph.batch

        #
        # GCN
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
        print("node_tokens:", graph.node_tokens.shape)
        print("node_types:", graph.node_types.shape)
        print("edge_index:", graph.edge_index.shape)
        print("batch:", graph.batch.shape)
        print("num graphs:", graph.num_graphs)
        print("batch max:", graph.batch.max().item())
        #
        # Graph embedding
        #
        x = global_max_pool(
            x,
            batch
        )

        #
        # Classification
        #
        logits = self.classifier(
            x
        )

        return logits