
from types import SimpleNamespace

from src.debug.pruning_overlap_diagnostics import (
    analyze_sample,
    jaccard_similarity,
    run_overlap_diagnostics,
)


def make_chain_sample():
    nodes = [
        SimpleNamespace(
            node_id=i,
            lineno=i,
            end_lineno=i,
            node_type="Statement",
            text="statement {}".format(i),
        )
        for i in range(1, 7)
    ]

    cfg = {
        "nodes": nodes,
        "edges": [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
        ],
    }

    return SimpleNamespace(
        cfg=cfg,
        function_nodes={1, 2, 3, 4, 5, 6},
        seed_nodes=[3],
        repo="synthetic",
        parent_commit="test-commit",
        file_path="example.py",
    )


def test_jaccard_similarity():
    assert jaccard_similarity({1, 2}, {2, 3}) == 1.0 / 3.0
    assert jaccard_similarity({1, 2}, {1, 2}) == 1.0
    assert jaccard_similarity({1}, {2}) == 0.0
    assert jaccard_similarity(set(), set()) == 0.0


def test_forward_backward_intersection():
    summary, pairs = analyze_sample(make_chain_sample())

    # Forward: {3, 4, 5, 6}
    # Backward: {1, 2, 3}
    assert summary["forward_backward_intersection_count"] == 1
    assert summary["forward_backward_union_count"] == 6
    assert summary["forward_backward_jaccard"] == 1.0 / 6.0

    assert summary["neighborhood_retention_ratio"] == 5.0 / 6.0
    assert summary["neighborhood_over_80_percent"] is True

    forward_backward = next(
        row for row in pairs
        if row["strategy_a"] == "forward"
        and row["strategy_b"] == "backward"
    )

    assert forward_backward["intersection_count"] == 1
    assert forward_backward["identical_node_sets"] is False
    assert forward_backward["nearly_identical"] is False


def test_all_pairwise_comparisons_are_returned():
    summary_rows, pairwise_rows = run_overlap_diagnostics(
        [make_chain_sample()]
    )

    assert len(summary_rows) == 1
    assert len(pairwise_rows) == 3

    assert {
        (row["strategy_a"], row["strategy_b"])
        for row in pairwise_rows
    } == {
        ("forward", "backward"),
        ("forward", "neighborhood"),
        ("backward", "neighborhood"),
    }
