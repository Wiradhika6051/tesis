# tesis/dataset/ast_graph_builder.py

import ast
import torch
from torch_geometric.data import Data


class ASTGraphBuilder(ast.NodeVisitor):

    def __init__(self, vocab):
        self.vocab = vocab

        self.nodes = []
        self.edges = []

        self.node_counter = 0

    def add_node(self, node):

        node_id = self.node_counter

        self.node_counter += 1

        node_type = type(node).__name__

        token_id = self.vocab.get(
            node_type,
            self.vocab["<UNK>"]
        )

        self.nodes.append(
            [token_id]
        )

        return node_id

    def visit_with_parent(
        self,
        node,
        parent_id=None
    ):

        current_id = self.add_node(
            node
        )

        if parent_id is not None:

            self.edges.append(
                [parent_id, current_id]
            )

            self.edges.append(
                [current_id, parent_id]
            )

        for child in ast.iter_child_nodes(node):

            self.visit_with_parent(
                child,
                current_id
            )

    def build(
        self,
        code
    ):

        try:

            tree = ast.parse(code)

        except Exception:

            return None

        self.visit_with_parent(tree)

        x = torch.tensor(
            self.nodes,
            dtype=torch.long
        )

        if len(self.edges) == 0:

            edge_index = torch.tensor(
                [[0], [0]],
                dtype=torch.long
            )

        else:

            edge_index = torch.tensor(
                self.edges,
                dtype=torch.long
            ).t().contiguous()

        return Data(
            x=x,
            edge_index=edge_index
        )


def build_ast_graph(
    code,
    vocab
):

    builder = ASTGraphBuilder(
        vocab
    )

    return builder.build(
        code
    )