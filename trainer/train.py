import torch

from torch_geometric.data import (
    Batch
)


def collate_fn(batch):

    graphs = [x[0] for x in batch]

    tokens = torch.stack(
        [x[1] for x in batch]
    )

    labels = torch.stack(
        [x[2] for x in batch]
    )

    graphs = Batch.from_data_list(
        graphs
    )

    return (
        graphs,
        tokens,
        labels
    )


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):

    model.train()

    total_loss = 0

    for graph, tokens, labels in loader:

        graph = graph.to(device)
        tokens = tokens.to(device)
        labels = labels.to(device)

        cfg_emb = model.encode_graph(
            graph
        )

        tok_emb = model.encode_tokens(
            tokens
        )

        logits = model(
            cfg_emb,
            tok_emb
        )

        loss = criterion(
            logits,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)