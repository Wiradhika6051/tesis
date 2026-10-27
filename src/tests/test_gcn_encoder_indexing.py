
from types import SimpleNamespace

import pytest
import torch

from src.encoding.gcn_encoder import GCNEncoder


class FakeVectorizer:
    """Return a small, deterministic vector sequence."""

    def text_to_vectors(self, text):
        if not text:
            return []

        return [
            [1.0, 2.0],
            [3.0, 4.0],
        ]


@pytest.fixture
def encoder():
    return GCNEncoder(
        vudenc_vectorizer=FakeVectorizer(),
        cfg_vocab={
            "<UNK>": 0,
            "Assign": 1,
            "Return": 2,
        },
    )


def make_node(node_id, text="x", node_type="Assign"):
    return SimpleNamespace(
        node_id=node_id,
        text=text,
        node_type=node_type,
    )


def make_sample(nodes, edges):
    return SimpleNamespace(
        pruned_cfg={
            "nodes": nodes,
            "edges": edges,
        },
        repo="test-repo",
        parent_commit="abc123",
        file_path="example.py",
        function_start=1,
        function_end=10,
        label=1,
        graph=None,
    )


def encode(encoder, nodes, edges):
    sample = make_sample(nodes, edges)
    encoder.encode(sample)
    return sample.graph


def test_non_contiguous_node_ids_are_remapped(encoder):
    graph = encode(
        encoder,
        nodes=[
            make_node(10),
            make_node(30),
            make_node(90, node_type="Return"),
        ],
        edges=[
            (10, 30),
            (30, 90),
        ],
    )

    assert graph.num_nodes == 3
    assert graph.edge_index.dtype == torch.long

    assert graph.edge_index.tolist() == [
        [0, 1],
        [1, 2],
    ]


def test_every_edge_endpoint_is_in_valid_range(encoder):
    graph = encode(
        encoder,
        nodes=[
            make_node(10),
            make_node(30),
            make_node(90),
        ],
        edges=[
            (10, 90),
            (90, 30),
        ],
    )

    num_nodes = graph.num_nodes

    assert graph.edge_index.shape[0] == 2

    if graph.edge_index.numel() > 0:
        assert torch.all(graph.edge_index >= 0)
        assert torch.all(graph.edge_index < num_nodes)


def test_empty_graph_is_supported(encoder):
    graph = encode(
        encoder,
        nodes=[],
        edges=[],
    )

    assert graph.num_nodes == 0
    assert graph.edge_index.shape == (2, 0)
    assert graph.node_types.shape == (0,)
    assert graph.node_token_vectors == []


def test_isolated_nodes_are_retained(encoder):
    graph = encode(
        encoder,
        nodes=[
            make_node(10),
            make_node(50),
            make_node(99),
        ],
        edges=[],
    )

    assert graph.num_nodes == 3
    assert graph.node_types.shape == (3,)
    assert len(graph.node_token_vectors) == 3
    assert graph.edge_index.shape == (2, 0)


def test_graph_with_nodes_but_no_edges(encoder):
    graph = encode(
        encoder,
        nodes=[
            make_node(7),
            make_node(42, node_type="Return"),
        ],
        edges=[],
    )

    assert graph.num_nodes == 2
    assert graph.edge_index.shape == (2, 0)
    assert graph.node_types.tolist() == [1, 2]


def test_missing_edge_endpoint_raises_value_error(encoder):
    sample = make_sample(
        nodes=[
            make_node(10),
            make_node(30),
        ],
        edges=[
            (10, 999),
        ],
    )

    with pytest.raises(
        ValueError,
        match="missing nodes",
    ):
        encoder.encode(sample)


def test_node_features_remain_aligned_with_indices(encoder):
    graph = encode(
        encoder,
        nodes=[
            make_node(90, node_type="Return"),
            make_node(10, node_type="Assign"),
        ],
        edges=[
            (90, 10),
        ],
    )

    # The node order determines the tensor-row order.
    assert graph.node_types.tolist() == [2, 1]
    assert len(graph.node_token_vectors) == 2

    # Original edge 90 -> 10 becomes tensor index 0 -> 1.
    assert graph.edge_index.tolist() == [[0], [1]]

def test_duplicate_node_ids_are_rejected(encoder):
    sample = make_sample(
        nodes=[
            make_node(10),
            make_node(10),
        ],
        edges=[],
    )

    with pytest.raises(ValueError, match="duplicate"):
        encoder.encode(sample)