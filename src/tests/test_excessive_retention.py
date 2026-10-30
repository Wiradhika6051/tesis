
from types import SimpleNamespace

from src.debug.investigate_excessive_retention import (
    investigate_sample,
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

    return SimpleNamespace(
        cfg={
            "nodes": nodes,
            "edges": [
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
            ],
        },
        function_nodes={1, 2, 3, 4, 5, 6},
        seed_nodes=[3],
        repo="synthetic",
        parent_commit="test",
        file_path="example.py",
    )


def test_hop_budget_sensitivity_and_retention():
    sample = make_chain_sample()

    original_edges = list(sample.cfg["edges"])
    original_ids = {
        node.node_id for node in sample.cfg["nodes"]
    }

    summary, hop_rows = investigate_sample(sample)

    by_hops = {
        row["hops"]: row for row in hop_rows
    }

    assert by_hops[0]["reachable_count"] == 1
    assert by_hops[1]["reachable_count"] == 3
    assert by_hops[2]["reachable_count"] == 5
    assert by_hops[3]["reachable_count"] == 6
    assert by_hops["unlimited"]["reachable_count"] == 6

    assert summary["function_node_count"] == 6
    assert summary["valid_seed_count"] == 1
    assert summary["neighborhood_hop2_retention_ratio"] == 5.0 / 6.0
    assert summary["high_retention"] is True

    # The original sample must remain unchanged.
    assert sample.cfg["edges"] == original_edges
    assert {
        node.node_id for node in sample.cfg["nodes"]
    } == original_ids
    assert not hasattr(sample, "pruned_cfg")


def test_multiple_seeds_increase_or_preserve_reachability():
    sample = make_chain_sample()

    one_seed, _ = investigate_sample(sample)

    sample.seed_nodes = [2, 5]
    two_seeds, _ = investigate_sample(sample)

    assert two_seeds["valid_seed_count"] == 2
    assert (
        two_seeds["neighborhood_hop2_retention_ratio"]
        >= one_seed["neighborhood_hop2_retention_ratio"]
    )
