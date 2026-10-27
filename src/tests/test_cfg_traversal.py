
from types import SimpleNamespace

import pytest

from src.pruners.cfg_traversal import traverse_cfg


def make_node(node_id, lineno):
    """Create a minimal CFG node.

    Non-positive line numbers represent virtual nodes.
    """
    return SimpleNamespace(
        node_id=node_id,
        lineno=lineno,
        end_lineno=lineno,
        node_type="virtual" if lineno <= 0 else "statement",
        text="",
    )


@pytest.fixture
def cfg():
    """
    Main function:
        1 -> 2 -> 3 -> 4
             |
             v
             5

    Virtual node 90 connects 2 to 3.

    Another function has nodes 6 -> 7.
    Node 8 is disconnected.
    """
    nodes = [
        make_node(1, 1),
        make_node(2, 2),
        make_node(90, 0),  # virtual node
        make_node(3, 3),
        make_node(4, 4),
        make_node(5, 5),
        make_node(6, 20),  # unrelated function
        make_node(7, 21),
        make_node(8, 30),  # disconnected node
    ]

    edges = [
        (1, 2),
        (2, 90),
        (90, 3),
        (3, 4),
        (3, 5),
        (6, 7),
    ]

    return {
        "nodes": nodes,
        "edges": edges,
    }


@pytest.fixture
def function_nodes():
    # Deliberately exclude nodes from the other function
    # and the disconnected node.
    return {1, 2, 90, 3, 4, 5}


def traverse(cfg, seeds, function_nodes, direction, hops=None):
    kwargs = {}

    if hops is not None:
        kwargs["max_ast_hops"] = hops

    return set(
        traverse_cfg(
            cfg=cfg,
            seed_nodes=seeds,
            function_nodes=function_nodes,
            direction=direction,
            **kwargs
        )
    )


# ---------------------------------------------------------
# 1. Direction-specific traversal
# ---------------------------------------------------------

def test_backward_traversal_follows_predecessors(
    cfg, function_nodes
):
    result = traverse(
        cfg, [3], function_nodes, "backward"
    )

    assert result == {1, 2, 90, 3}


def test_forward_traversal_follows_successors(
    cfg, function_nodes
):
    result = traverse(
        cfg, [3], function_nodes, "forward"
    )

    assert result == {3, 4, 5}


def test_neighborhood_traversal_is_undirected(
    cfg, function_nodes
):
    result = traverse(
        cfg, [3], function_nodes, "undirected"
    )

    assert result == {1, 2, 90, 3, 4, 5}


# ---------------------------------------------------------
# 2. Hop-budget handling
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "direction, expected",
    [
        ("backward", {3, 90}),
        ("forward", {3}),
        ("undirected", {3, 90}),
    ],
)
def test_zero_hops_keeps_only_seed(
    cfg, function_nodes, direction, expected
):
    result = traverse(
        cfg,
        [3],
        function_nodes,
        direction,
        hops=0,
    )

    assert result == expected


def test_forward_hop_budget_excludes_distant_nodes(
    cfg, function_nodes
):
    result = traverse(
        cfg, [3], function_nodes, "forward", hops=1
    )

    assert result == {3, 4, 5}


def test_backward_hop_budget_excludes_distant_nodes(
    cfg, function_nodes
):
    result = traverse(
        cfg, [3], function_nodes, "backward", hops=1
    )

    # Node 90 is virtual. Under the zero-cost virtual-node
    # contract, it is traversable without consuming an AST hop.
    assert result == {2, 3, 90}


def test_neighborhood_hop_budget(
    cfg, function_nodes
):
    result = traverse(
        cfg, [3], function_nodes, "undirected", hops=1
    )

    assert result == {2, 3, 4, 5, 90}


# ---------------------------------------------------------
# 3. Virtual-node traversal
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "direction, seed, expected",
    [
        ("forward", 2, {2, 3, 90}),
        ("backward", 3, {2, 3, 90}),
        ("undirected", 3, {2, 3, 4, 5, 90}),
    ],
)
def test_virtual_node_does_not_block_traversal(
    cfg, function_nodes, direction, seed, expected
):
    result = traverse(
        cfg,
        [seed],
        function_nodes,
        direction,
        hops=1,
    )

    assert result == expected


# ---------------------------------------------------------
# 4. Function-scope restrictions
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "direction",
    ["backward", "forward", "undirected"],
)
def test_traversal_does_not_leak_outside_function_scope(
    cfg, direction
):
    # Include the unrelated nodes in the CFG, but exclude
    # them from the allowed function scope.
    allowed = {1, 2, 90, 3, 4, 5}

    result = traverse(
        cfg, [3], allowed, direction
    )

    assert result <= allowed
    assert 6 not in result
    assert 7 not in result
    assert 8 not in result


def test_out_of_scope_seed_is_not_traversed(
    cfg, function_nodes
):
    result = traverse(
        cfg, [6], function_nodes, "forward"
    )

    assert result == set()


# ---------------------------------------------------------
# 5. Multiple seeds and node validity
# ---------------------------------------------------------

def test_multiple_seeds_return_union_of_reachable_nodes(
    cfg, function_nodes
):
    result = traverse(
        cfg, [1, 4], function_nodes, "forward"
    )

    assert result == {1, 2, 90, 3, 4, 5}


@pytest.mark.parametrize(
    "direction",
    ["backward", "forward", "undirected"],
)
def test_result_contains_only_existing_node_ids(
    cfg, function_nodes, direction
):
    result = traverse(
        cfg, [3], function_nodes, direction
    )

    actual_ids = {
        node.node_id for node in cfg["nodes"]
    }

    assert result <= actual_ids
    assert result <= function_nodes


def test_disconnected_node_is_not_included(
    cfg, function_nodes
):
    result = traverse(
        cfg, [3], function_nodes, "undirected"
    )

    assert 8 not in result


# ---------------------------------------------------------
# 6. Invalid direction
# ---------------------------------------------------------

def test_invalid_direction_is_rejected(
    cfg, function_nodes
):
    with pytest.raises(ValueError):
        traverse(
            cfg, [3], function_nodes, "sideways"
        )


def test_duplicate_seeds_do_not_change_result(cfg, function_nodes):
    result = traverse(
        cfg, [3, 3, 3], function_nodes, "forward"
    )
    assert result == {3, 4, 5}


def test_empty_seeds_return_empty_set(cfg, function_nodes):
    result = traverse(cfg, [], function_nodes, "forward")
    assert result == set()


def test_seed_missing_from_cfg_is_ignored(cfg, function_nodes):
    # ID 999 is allowed by the supplied scope set,
    # but does not exist in the actual CFG.
    result = traverse(cfg, [999], function_nodes | {999}, "forward")
    assert result == set()


def test_edges_with_missing_node_ids_are_ignored(function_nodes):
    cfg = {
        "nodes": [
            make_node(1, 1),
            make_node(2, 2),
        ],
        "edges": [
            (1, 2),
            (2, 999),
            (888, 1),
        ],
    }

    result = traverse(cfg, [1], {1, 2}, "forward")
    assert result == {1, 2}


def test_backward_traversal_does_not_include_successors(
    cfg, function_nodes
):
    result = traverse(cfg, [3], function_nodes, "backward")
    assert 4 not in result
    assert 5 not in result


def test_forward_traversal_does_not_include_predecessors(
    cfg, function_nodes
):
    result = traverse(cfg, [3], function_nodes, "forward")
    assert 1 not in result
    assert 2 not in result


def test_scope_boundary_blocks_path_to_other_function():
    cfg = {
        "nodes": [
            make_node(1, 1),
            make_node(2, 2),
            make_node(3, 20),
            make_node(4, 21),
        ],
        "edges": [
            (1, 2),
            (2, 3),
            (3, 4),
        ],
    }

    # Only nodes 1 and 2 belong to the target function.
    allowed = {1, 2}

    result = traverse(cfg, [1], allowed, "forward")

    assert result == {1, 2}
    assert 3 not in result
    assert 4 not in result
