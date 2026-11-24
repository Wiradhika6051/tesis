
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_max_pool


class PhpNetGraph(nn.Module):
    """
    DeepTective-scale PHP vulnerability classifier.

    Modes
    -----
    gcn_only:
        CFG node-type embeddings -> GCN -> global max pooling -> classifier.

    gru_only:
        Per-node token sequences -> BiGRU -> global max pooling -> classifier.

    combined:
        Per-node token sequences -> BiGRU
        CFG node types -> embedding
        Concatenate node features -> GCN -> global max pooling -> classifier.

    All modes return binary-class logits with shape [batch_size, 2].
    """

    VALID_MODES = {"gcn_only", "gru_only", "combined"}

    def __init__(self, cfg_vocab_size, word2vec_dim=200, mode="combined"):
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode {mode!r}. Expected one of "
                f"{sorted(self.VALID_MODES)}."
            )

        self.mode = mode
        self.word2vec_dim = word2vec_dim

        # DeepTective-scale dimensions.
        self.cfg_embedding_dim = 100
        self.gru_hidden_size = 200
        self.gru_num_layers = 2
        self.token_feature_dim = self.gru_hidden_size * 2

        # CFG node-type embedding.
        self.cfg_embedding = nn.Embedding(
            num_embeddings=cfg_vocab_size,
            embedding_dim=self.cfg_embedding_dim,
        )

        # Encode the token sequence belonging to each CFG node.
        self.gru = nn.GRU(
            input_size=word2vec_dim,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Node features:
        # gcn_only: 100
        # combined: 400 token + 100 CFG type = 500
        if mode == "gcn_only":
            gcn_input_dim = self.cfg_embedding_dim
        else:
            gcn_input_dim = (
                self.token_feature_dim + self.cfg_embedding_dim
            )

        # DeepTective-scale graph encoder.
        self.conv1 = GCNConv(gcn_input_dim, 2000)
        self.conv2 = GCNConv(2000, 4000)
        self.conv3 = GCNConv(4000, 4000)

        # The pooled graph representation is 4000-dimensional.
        # GRU-only instead produces a 400-dimensional representation.
        classifier_input_dim = (
            self.token_feature_dim if mode == "gru_only" else 4000
        )

        # DeepTective-inspired classification head, adapted to two classes.
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 1000),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(500, 2),
        )

    def encode_node(self, node_vectors):
        """
        Encode one CFG node's token vectors.

        Input:
            [num_tokens, word2vec_dim]

        Output:
            [400]
        """
        if node_vectors.size(0) == 0:
            return torch.zeros(
                self.token_feature_dim,
                dtype=self.gru.weight_ih_l0.dtype,
                device=node_vectors.device,
            )

        node_vectors = node_vectors.unsqueeze(0)
        _, hidden = self.gru(node_vectors)

        # Last-layer forward and backward hidden states.
        forward = hidden[-2, 0]
        backward = hidden[-1, 0]

        return torch.cat((forward, backward), dim=0)

    def build_token_features(self, graph):
        """Return one 400-dimensional vector per CFG node."""
        token_features = []

        for node_vectors in graph.node_token_vectors:
            node_vectors = node_vectors.to(
                device=graph.node_types.device,
                dtype=self.gru.weight_ih_l0.dtype,
            )
            token_features.append(self.encode_node(node_vectors))

        if not token_features:
            raise ValueError("Graph batch contains no CFG nodes.")

        return torch.stack(token_features, dim=0)

    def build_node_features(self, graph):
        """Concatenate token representations and CFG-type embeddings."""
        token_features = self.build_token_features(graph)
        cfg_features = self.cfg_embedding(graph.node_types)

        return torch.cat((token_features, cfg_features), dim=1)

    def encode_graph(self, x, graph):
        """Apply the GCN and return a 4000-dimensional graph vector."""
        x = F.relu(self.conv1(x, graph.edge_index))
        x = F.relu(self.conv2(x, graph.edge_index))
        x = F.relu(self.conv3(x, graph.edge_index))

        return global_max_pool(x, graph.batch)

    def forward(self, graph):
        if self.mode == "gru_only":
            # Token representations without CFG message passing.
            x = self.build_token_features(graph)
            x = global_max_pool(x, graph.batch)

        elif self.mode == "gcn_only":
            # CFG node types and edges without token representations.
            x = self.cfg_embedding(graph.node_types)
            x = self.encode_graph(x, graph)

        else:
            # Early fusion at each CFG node.
            x = self.build_node_features(graph)
            x = self.encode_graph(x, graph)

        # Return raw logits for CrossEntropyLoss.
        return self.classifier(x)
