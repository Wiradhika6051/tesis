import torch

from torch_geometric.data import Data


def build_token_graph(
    token_ids
):

    if not token_ids:
        token_ids = [0]

    x = torch.tensor(
        token_ids,
        dtype=torch.long
    ).view(-1, 1)

    edges = []

    for i in range(
        len(token_ids) - 1
    ):

        edges.append(
            [i, i + 1]
        )

        edges.append(
            [i + 1, i]
        )

    if edges:

        edge_index = torch.tensor(
            edges,
            dtype=torch.long
        ).t()

    else:

        edge_index = torch.tensor(
            [[0], [0]],
            dtype=torch.long
        )

    return Data(
        x=x,
        edge_index=edge_index
    )