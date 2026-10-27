
from types import SimpleNamespace

import pytest

from src.pruners.backward_slice_pruner import BackwardSlicePruner
from src.pruners.forward_slice_pruner import ForwardSlicePruner
from src.pruners.neighborhood_pruner import NeighborhoodPruner


def make_node(node_id, lineno):
    return SimpleNamespace(
        node_id=node_id,
        lineno=lineno,
        end_lineno=lineno,
        node_type="virtual" if lineno <= 0 else "statement",
        text="",
    )


@pytest.fixture
def sample():
    cfg = {
        "nodes": [
            make_node(1, 1),
            make_node(2, 2),
            make_node(90, 0),
            make_node(3, 3),
            make_node(4, 4),
            make_node(5, 5),
            make_node(6, 20),
        ],
        "edges": [
            (1, 2),
            (2, 90),
            (90, 3),
            (3, 4),
            (3, 5),
            (5, 6),
        ],
    }

    return SimpleNamespace(
        cfg=cfg,
        seed_nodes=[3],
        function_nodes={1, 2, 90, 3, 4, 5},
        pruned_cfg=None,
    )


def retained_ids(sample):
    return {
        node.node_id
        for node in sample.pruned_cfg["nodes"]
    }


def assert_valid_pruned_cfg(sample):
    original_ids = {
        node.node_id
        for node in sample.cfg["nodes"]
    }

    pruned_ids = retained_ids(sample)

    assert pruned_ids <= original_ids
    assert pruned_ids <= sample.function_nodes

    for src, dst in sample.pruned_cfg["edges"]:
        assert src in pruned_ids
        assert dst in pruned_ids


def test_backward_pruner(sample):
    result = BackwardSlicePruner().prune(sample)

    assert retained_ids(result) == {1, 2, 90, 3}
    assert_valid_pruned_cfg(result)


def test_forward_pruner(sample):
    result = ForwardSlicePruner().prune(sample)

    assert retained_ids(result) == {3, 4, 5}
    assert_valid_pruned_cfg(result)


def test_neighborhood_pruner(sample):
    result = NeighborhoodPruner(hops=1).prune(sample)

    assert retained_ids(result) == {2, 90, 3, 4, 5}
    assert_valid_pruned_cfg(result)
