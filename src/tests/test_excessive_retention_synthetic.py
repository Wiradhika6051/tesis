
from types import SimpleNamespace

import pytest

from src.pruners.cfg_traversal import traverse_cfg


def make_node(node_id, lineno=None):
    """Create a minimal CFG node compatible with the current traversal."""
    if lineno is None:
        lineno = node_id

    return SimpleNamespace(
        node_id=node_id,
        lineno=lineno,
        node_type="SyntheticNode",
        text=f"node_{node_id}",
    )


def make_cfg(node_ids, edges, virtual_ids=()):
    """Build a synthetic CFG using the project's node representation."""
    virtual_ids = set(virtual_ids)

    return {
        "nodes": [
            make_node(node_id, lineno=-1 if node_id in virtual_ids else node_id)
            for node_id in node_ids
        ],
        "edges": list(edges),
    }


def run_neighborhood(cfg, seeds, function_nodes, hops):
    return traverse_cfg(
        cfg=cfg,
        seed_nodes=set(seeds),
        function_nodes=set(function_nodes),
        direction="undirected",
        max_ast_hops=hops,
    )


def test_seed_density_can_increase_retention():
    """More seeds should not reduce the reachable set for the same graph."""
    cfg = make_cfg(
        node_ids=range(1, 8),
        edges=[(i, i + 1) for i in range(1, 7)],
    )
    function_nodes = set(range(1, 8))

    one_seed = run_neighborhood(
        cfg, seeds={3}, function_nodes=function_nodes, hops=1
    )
    multiple_seeds = run_neighborhood(
        cfg, seeds={2, 3, 4, 5}, function_nodes=function_nodes, hops=1
    )

    assert one_seed == {2, 3, 4}
    assert multiple_seeds == {1, 2, 3, 4, 5, 6}
    assert one_seed <= multiple_seeds
    assert len(multiple_seeds) > len(one_seed)


def test_connectivity_can_increase_retention():
    """A star can reach more nodes than a chain at the same hop budget."""
    node_ids = range(1, 8)
    function_nodes = set(node_ids)

    chain = make_cfg(
        node_ids=node_ids,
        edges=[(i, i + 1) for i in range(1, 7)],
    )
    star = make_cfg(
        node_ids=node_ids,
        edges=[(1, i) for i in range(2, 8)],
    )

    chain_result = run_neighborhood(
        chain, seeds={1}, function_nodes=function_nodes, hops=1
    )
    star_result = run_neighborhood(
        star, seeds={1}, function_nodes=function_nodes, hops=1
    )

    assert chain_result == {1, 2}
    assert star_result == set(node_ids)
    assert len(star_result) > len(chain_result)


@pytest.mark.parametrize(
    ("hops", "expected"),
    [
        (0, {3}),
        (1, {2, 3, 4}),
        (2, {1, 2, 3, 4, 5}),
        (3, {1, 2, 3, 4, 5, 6}),
        (None, {1, 2, 3, 4, 5, 6}),
    ],
)
def test_hop_budget_has_expected_effect(hops, expected):
    """Verify the exact reachable set at each hop budget."""
    cfg = make_cfg(
        node_ids=range(1, 7),
        edges=[(i, i + 1) for i in range(1, 6)],
    )

    result = run_neighborhood(
        cfg,
        seeds={3},
        function_nodes=set(range(1, 7)),
        hops=hops,
    )

    assert result == expected


def test_traversal_does_not_escape_function_scope():
    """An ordinary AST node outside the function scope is a boundary."""
    cfg = make_cfg(
        node_ids=[1, 2, 3, 4],
        edges=[(1, 2), (2, 3), (3, 4)],
    )

    result = run_neighborhood(
        cfg,
        seeds={1},
        function_nodes={1, 2},
        hops=None,
    )

    assert result == {1, 2}
    assert 3 not in result
    assert 4 not in result


def test_virtual_nodes_do_not_consume_ast_hops():
    """Virtual nodes can connect AST nodes without costing AST hops."""
    cfg = make_cfg(
        node_ids=[1, 100, 101, 2],
        edges=[(1, 100), (100, 101), (101, 2)],
        virtual_ids={100, 101},
    )

    # Virtual nodes are intermediates, not ordinary function AST nodes.
    result = run_neighborhood(
        cfg,
        seeds={1},
        function_nodes={1, 2},
        hops=1,
    )

    assert result == {1, 2}


def test_empty_graph_returns_empty_set():
    cfg = make_cfg(node_ids=[], edges=[])

    result = run_neighborhood(
        cfg, seeds=set(), function_nodes=set(), hops=2
    )

    assert result == set()


def test_invalid_seed_does_not_enter_result():
    cfg = make_cfg(
        node_ids=[1, 2, 3],
        edges=[(1, 2), (2, 3)],
    )

    result = run_neighborhood(
        cfg,
        seeds={999},
        function_nodes={1, 2, 3},
        hops=2,
    )

    assert result == set()


def test_traversal_does_not_mutate_original_cfg():
    cfg = make_cfg(
        node_ids=[1, 2, 3],
        edges=[(1, 2), (2, 3)],
    )
    original_nodes = list(cfg["nodes"])
    original_edges = list(cfg["edges"])

    run_neighborhood(
        cfg,
        seeds={2},
        function_nodes={1, 2, 3},
        hops=1,
    )

    assert cfg["nodes"] == original_nodes
    assert cfg["edges"] == original_edges
