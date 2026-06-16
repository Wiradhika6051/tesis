import torch
from torch_geometric.data import Data


def build_token_graph(token_ids):
    """
    Sequential token graph

    token0 <-> token1 <-> token2 ...
    """

    if len(token_ids) == 0:
        token_ids = [0]

    x = torch.tensor(
        token_ids,
        dtype=torch.long
    ).view(-1, 1)

    edges = []

    for i in range(len(token_ids) - 1):

        edges.append([i, i + 1])
        edges.append([i + 1, i])

    if len(edges) == 0:

        edge_index = torch.tensor(
            [[0], [0]],
            dtype=torch.long
        )

    else:

        edge_index = torch.tensor(
            edges,
            dtype=torch.long
        ).t().contiguous()

    return Data(
        x=x,
        edge_index=edge_index
    )