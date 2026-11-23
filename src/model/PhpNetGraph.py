import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_max_pool


class PhpNetGraph(nn.Module):
    """
    PHP vulnerability classifier with three ablation modes.

    Modes
    -----
    gcn_only:
        CFG node-type embeddings -> GCN -> global max pooling -> classifier.
        Ignores token vectors.

    gru_only:
        Word2Vec tokens are encoded independently within each CFG node using
        the BiGRU. Node vectors are pooled per graph without message passing.
        Ignores CFG node-type embeddings and CFG edges.

    combined:
        Original architecture: concatenate each node's BiGRU representation
        and CFG node-type embedding, then apply GCN and graph pooling.

    All modes return two-class logits with shape [batch_size, 2].
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

        # CFG node-type embedding (used by GCN-only and combined modes).
        self.cfg_embedding = nn.Embedding(
            num_embeddings=cfg_vocab_size,
            embedding_dim=32,
        )

        # Encode the token sequence belonging to each CFG node.
        self.gru = nn.GRU(
            input_size=word2vec_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # GCN input dimension depends on whether token features are included.
        gcn_input_dim = 32 if mode == "gcn_only" else 160

        self.conv1 = GCNConv(gcn_input_dim, 128)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 256)

        # GRU-only pools 128-dimensional node vectors; other modes pool 256.
        classifier_input_dim = 128 if mode == "gru_only" else 256

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def encode_node(self, node_vectors):
        """
        Encode one CFG node's token vectors.

        Parameters
        ----------
        node_vectors : Tensor
            Shape [num_tokens, word2vec_dim].

        Returns
        -------
        Tensor
            Shape [128], concatenating the last-layer forward and backward
            GRU hidden states.
        """
        if node_vectors.size(0) == 0:
            return torch.zeros(
                128,
                dtype=torch.float32,
                device=node_vectors.device,
            )

        node_vectors = node_vectors.unsqueeze(0)
        _, hidden = self.gru(node_vectors)

        forward = hidden[-2, 0]
        backward = hidden[-1, 0]

        return torch.cat((forward, backward), dim=0)

    def build_token_features(self, graph):
        """Return one 128-dimensional BiGRU vector per CFG node."""
        token_features = []

        for node_vectors in graph.node_token_vectors:
            node_vectors = node_vectors.to(graph.node_types.device)
            token_features.append(self.encode_node(node_vectors))

        if not token_features:
            raise ValueError("Graph batch contains no CFG nodes.")

        return torch.stack(token_features, dim=0)

    def build_node_features(self, graph):
        """Concatenate each node's BiGRU and CFG-type representations."""
        token_features = self.build_token_features(graph)
        cfg_features = self.cfg_embedding(graph.node_types)

        return torch.cat((token_features, cfg_features), dim=1)

    def forward(self, graph):
        if self.mode == "gru_only":
            # No GCN and no CFG type embedding: pool token-derived node
            # representations directly for each graph in the batch.
            x = self.build_token_features(graph)
            x = global_max_pool(x, graph.batch)

        elif self.mode == "gcn_only":
            # No GRU/token features: use CFG node types and graph edges only.
            x = self.cfg_embedding(graph.node_types)

            x = F.relu(self.conv1(x, graph.edge_index))
            x = F.relu(self.conv2(x, graph.edge_index))
            x = F.relu(self.conv3(x, graph.edge_index))

            x = global_max_pool(x, graph.batch)

        else:  # combined: original architecture
            x = self.build_node_features(graph)

            x = F.relu(self.conv1(x, graph.edge_index))
            x = F.relu(self.conv2(x, graph.edge_index))
            x = F.relu(self.conv3(x, graph.edge_index))

            x = global_max_pool(x, graph.batch)

        return self.classifier(x)
