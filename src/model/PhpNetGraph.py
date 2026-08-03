import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    EdgePooling,
    global_max_pool
)


class PhpNetGraph(nn.Module):

    def __init__(
        self,
        token_vocab_size
    ):

        super().__init__()

        #
        # Token embedding
        #
        self.embedding = nn.Embedding(
            token_vocab_size,
            100
        )

        #
        # Encode every CFG node independently.
        #
        self.gru = nn.GRU(
            input_size=100,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        #
        # GCN
        #
        self.conv1 = GCNConv(
            128,
            128
        )

        self.pool1 = EdgePooling(
            128
        )

        self.conv2 = GCNConv(
            128,
            256
        )

        self.pool2 = EdgePooling(
            256
        )

        self.conv3 = GCNConv(
            256,
            256
        )

        self.pool3 = EdgePooling(
            256
        )

        #
        # Classification
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

    def encode_nodes(
        self,
        node_tokens
    ):

        #
        # node_tokens
        #
        # (num_nodes,
        #  max_tokens_per_node)
        #

        x = self.embedding(
            node_tokens
        )

        _, hidden = self.gru(
            x
        )

        #
        # Forward + backward
        #
        node_embedding = torch.cat(

            (
                hidden[-2],
                hidden[-1]
            ),

            dim=1

        )

        return node_embedding

    def forward(
        self,
        graph
    ):

        x = self.encode_nodes(
            graph.x.long()
        )

        edge_index = graph.edge_index

        batch = graph.batch

        x = self.conv1(
            x,
            edge_index
        )

        x, edge_index, batch, _ = self.pool1(
            x,
            edge_index,
            batch=batch
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x, edge_index, batch, _ = self.pool2(
            x,
            edge_index,
            batch=batch
        )

        x = F.relu(x)

        x = self.conv3(
            x,
            edge_index
        )

        x, edge_index, batch, _ = self.pool3(
            x,
            edge_index,
            batch=batch
        )

        x = F.relu(x)

        x = global_max_pool(
            x,
            batch
        )

        return self.classifier(
            x
        )