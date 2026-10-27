"""Unit tests for multi-seed CFG traversal and pruning.

Place this file in the repository's tests directory (for example,
``src/tests/test_cfg_pruning.py``) and run it with pytest.

The imports below follow the module paths used by the uploaded pruning files.
If the repository has since consolidated all pruners under ``src.pruners``,
change only the four pruner import statements accordingly.
"""
from types import SimpleNamespace

import pytest

from src.cfg.cfg_node import CFGNode
from src.pruners.cfg_traversal import traverse_cfg
from src.pruners.utils import prune_cfg_preserving_virtual_paths
from src.pruners.forward_slice_pruner import ForwardSlicePruner
from src.pruners.backward_slice_pruner import BackwardSlicePruner
from src.pruners.neighborhood_pruner import NeighborhoodPruner
from src.pruners.identity_pruner import IdentityPruner


def make_node(node_id, lineno, node_type="Statement", end_lineno=None):
    return CFGNode(
        node_id=node_id,
        lineno=lineno,
        end_lineno=end_lineno,
        node_type=node_type,
        text="node_{}".format(node_id),
    )


def make_cfg():
    """A small CFG with a virtual node and two branches.

    AST nodes: 1 -> 2 -> 3 -> 4
                  \-> 5
    Virtual node 10 sits between nodes 2 and 3.
    """
    nodes = [
        make_node(1, 1, "FunctionDef"),
        make_node(2, 2, "Assign"),
        make_node(3, 3, "If"),
        make_node(4, 4, "Return"),
        make_node(5, 5, "Assign"),
        make_node(10, -1, "VIRTUAL"),
    ]
    edges = [(1, 2), (2, 10), (10, 3), (3, 4), (2, 5)]
    return {"nodes": nodes, "edges": edges}


def make_sample(seed_nodes, function_nodes=None, cfg=None):
    cfg = cfg or make_cfg()
    if function_nodes is None:
        function_nodes = {1, 2, 3, 4, 5}
    return SimpleNamespace(
        cfg=cfg,
        seed_nodes=list(seed_nodes),
        function_nodes=set(function_nodes),
        pruned_cfg=None,
    )


def node_ids(cfg):
    return {node.node_id for node in cfg["nodes"]}


# ---------------------------------------------------------------------------
# CFG traversal: multi-seed behavior
# ---------------------------------------------------------------------------


def test_forward_traversal_combines_reachability_from_all_seeds():
    result = traverse_cfg(
        cfg=make_cfg(),
        seed_nodes=[1, 5],
        function_nodes={1, 2, 3, 4, 5},
        direction="forward",
    )
    assert result == {1, 2, 3, 4, 5}


def test_backward_traversal_combines_reachability_from_all_seeds():
    result = traverse_cfg(
        cfg=make_cfg(),
        seed_nodes=[4, 5],
        function_nodes={1, 2, 3, 4, 5},
        direction="backward",
    )
    assert result == {1, 2, 3, 4, 5}


def test_undirected_traversal_combines_multiple_seeds():
    result = traverse_cfg(
        cfg=make_cfg(),
        seed_nodes=[4, 5],
        function_nodes={1, 2, 3, 4, 5},
        direction="undirected",
    )
    assert result == {1, 2, 3, 4, 5}


def test_virtual_nodes_are_traversed_but_not_returned():
    result = traverse_cfg(
        cfg=make_cfg(),
        seed_nodes=[2],
        function_nodes={1, 2, 3, 4, 5},
        direction="forward",
    )
    assert result == {2, 3, 4, 5}
    assert 10 not in result


def test_nodes_outside_target_function_are_traversal_boundaries():
    cfg = make_cfg()
    cfg["nodes"].append(make_node(20, 20, "OtherFunction"))
    cfg["edges"].append((4, 20))

    result = traverse_cfg(
        cfg=cfg,
        seed_nodes=[1],
        function_nodes={1, 2, 3, 4, 5},
        direction="forward",
    )
    assert result == {1, 2, 3, 4, 5}
    assert 20 not in result


def test_invalid_seed_ids_are_ignored_when_valid_seed_exists():
    result = traverse_cfg(
        cfg=make_cfg(),
        seed_nodes=[999, 1],
        function_nodes={1, 2, 3, 4, 5},
        direction="forward",
    )
    assert result == {1, 2, 3, 4, 5}


def test_no_seed_in_target_function_returns_empty_set():
    result = traverse_cfg(
        cfg=make_cfg(),
        seed_nodes=[999],
        function_nodes={1, 2, 3, 4, 5},
        direction="forward",
    )
    assert result == set()


def test_unsupported_traversal_direction_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported direction"):
        traverse_cfg(
            cfg=make_cfg(),
            seed_nodes=[1],
            function_nodes={1, 2, 3, 4, 5},
            direction="sideways",
        )


def test_neighborhood_hop_limit_counts_ast_nodes_not_virtual_nodes():
    result = traverse_cfg(
        cfg=make_cfg(),
        seed_nodes=[2],
        function_nodes={1, 2, 3, 4, 5},
        direction="undirected",
        max_ast_hops=1,
    )
    # Node 3 is one AST hop away through virtual node 10; node 10 itself
    # is not returned and does not consume an AST hop.
    assert result == {1, 2, 3, 5}


# ---------------------------------------------------------------------------
# CFG pruning utility: node retention and virtual-path contraction
# ---------------------------------------------------------------------------


def test_prune_cfg_preserving_virtual_paths_contracts_virtual_nodes():
    cfg = make_cfg()
    pruned = prune_cfg_preserving_virtual_paths(cfg, {2, 3})

    assert node_ids(pruned) == {2, 3}
    assert (2, 3) in pruned["edges"]
    assert all(src != 10 and dst != 10 for src, dst in pruned["edges"])


def test_prune_cfg_does_not_contract_through_removed_ast_nodes():
    cfg = make_cfg()
    pruned = prune_cfg_preserving_virtual_paths(cfg, {1, 3})

    assert node_ids(pruned) == {1, 3}
    # The only route from 1 to 3 crosses retained/removed AST node 2,
    # so pruning must not invent a direct edge between 1 and 3.
    assert (1, 3) not in pruned["edges"]


def test_prune_cfg_with_empty_keep_set_returns_empty_graph():
    pruned = prune_cfg_preserving_virtual_paths(make_cfg(), set())
    assert pruned["nodes"] == []
    assert pruned["edges"] == []


# ---------------------------------------------------------------------------
# Individual pruner classes: multiple seed nodes and edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pruner,direction",
    [
        (ForwardSlicePruner(), "forward"),
        (BackwardSlicePruner(), "backward"),
        (NeighborhoodPruner(hops=2), "undirected"),
    ],
)
def test_traversal_pruners_accept_multiple_seed_nodes(pruner, direction):
    sample = make_sample(seed_nodes=[2, 4])
    result = pruner.prune(sample)

    assert result is sample
    assert result.pruned_cfg is not None
    # Every valid seed is retained by traversal-based pruning.
    assert {2, 4}.issubset(node_ids(result.pruned_cfg))
    # Virtual nodes are not part of the retained AST-node set.
    assert 10 not in node_ids(result.pruned_cfg)


def test_forward_pruner_keeps_union_of_forward_slices():
    sample = make_sample(seed_nodes=[1, 5])
    ForwardSlicePruner().prune(sample)
    assert node_ids(sample.pruned_cfg) == {1, 2, 3, 4, 5}


def test_backward_pruner_keeps_union_of_backward_slices():
    sample = make_sample(seed_nodes=[4, 5])
    BackwardSlicePruner().prune(sample)
    assert node_ids(sample.pruned_cfg) == {1, 2, 3, 4, 5}


def test_neighborhood_pruner_honors_configured_hop_limit():
    sample = make_sample(seed_nodes=[2])
    NeighborhoodPruner(hops=1).prune(sample)
    assert node_ids(sample.pruned_cfg) == {1, 2, 3, 5}


def test_identity_pruner_keeps_all_function_nodes_not_virtual_nodes():
    sample = make_sample(seed_nodes=[2])
    IdentityPruner().prune(sample)
    assert node_ids(sample.pruned_cfg) == {1, 2, 3, 4, 5}
    assert 10 not in node_ids(sample.pruned_cfg)


@pytest.mark.parametrize(
    "pruner",
    [
        ForwardSlicePruner(),
        BackwardSlicePruner(),
        NeighborhoodPruner(hops=2),
    ],
)
def test_traversal_pruners_return_empty_graph_when_no_seed_is_valid(pruner):
    sample = make_sample(seed_nodes=[999])
    pruner.prune(sample)
    assert sample.pruned_cfg["nodes"] == []
    assert sample.pruned_cfg["edges"] == []


def test_identity_pruner_does_not_depend_on_seed_nodes():
    sample = make_sample(seed_nodes=[])
    IdentityPruner().prune(sample)
    assert node_ids(sample.pruned_cfg) == {1, 2, 3, 4, 5}


def test_pruner_does_not_mutate_original_cfg():
    sample = make_sample(seed_nodes=[2])
    original_node_ids = node_ids(sample.cfg)
    original_edges = list(sample.cfg["edges"])

    ForwardSlicePruner().prune(sample)

    assert node_ids(sample.cfg) == original_node_ids
    assert sample.cfg["edges"] == original_edges
