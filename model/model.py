import torch
import torch.nn as nn

class TokenGRU(nn.Module):

    def __init__(
        self,
        vocab_size
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            128,
            padding_idx=0
        )

        self.gru = nn.GRU(
            128,
            128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                2
            )
        )

    def forward(
        self,
        tokens
    ):

        x = self.embedding(tokens)

        _, hidden = self.gru(x)

        hidden = torch.cat(
            [hidden[i]
             for i in range(hidden.shape[0])],
            dim=1
        )

        return self.classifier(hidden)