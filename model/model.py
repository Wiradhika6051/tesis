# model.py

import torch
import torch.nn as nn


class TokenGRU(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, tokens):

        x = self.embedding(tokens)

        _, hidden = self.gru(x)

        hidden = torch.cat(
            [hidden[i] for i in range(hidden.shape[0])],
            dim=1
        )

        return self.classifier(hidden)