# src/tests/test_pruning_diagnostics.py

from types import SimpleNamespace

from src.debug.pruning_diagnostics import run_pruning_diagnostics


def make_sample():
    # Directed CFG:
    #
    # 1 -> 2 -> 3 -> 4 -> 5 -> 6
    #
    # The same seed (3) is used for every strategy.
    nodes = [
        SimpleNamespace(
            node_id=node_id,
            node_type="Statement",
            lineno=node_id,
            text="statement_{}".format(node_id),
        )
        for node_id in range(1, 7)
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
        repo="test-repo",
        parent_commit="abc123",
        file_path="example.py",
        function_start=1,
        function_end=6,
        label=1,
        pruned_cfg=None,
    )


def test_all_strategies_use_same_sample_and_seed_count():
    sample = make_sample()

    rows = run_pruning_diagnostics(
        [sample],
        neighborhood_hops=2,
    )

    assert len(rows) == 3

    assert {row["strategy"] for row in rows} == {
        "forward",
        "backward",
        "neighborhood",
    }

    assert all(row["original_node_count"] == 6 for row in rows)
    assert all(row["seed_count"] == 1 for row in rows)
    assert all(row["valid_seed_count"] == 1 for row in rows)


def test_expected_retention_for_simple_chain():
    rows = run_pruning_diagnostics(
        [make_sample()],
        neighborhood_hops=2,
    )

    by_strategy = {
        row["strategy"]: row
        for row in rows
    }

    # Forward from node 3: 3, 4, 5, 6
    assert by_strategy["forward"]["retained_node_count"] == 4

    # Backward from node 3: 1, 2, 3
    assert by_strategy["backward"]["retained_node_count"] == 3

    # Undirected distance <= 2 from node 3: 1, 2, 3, 4, 5
    assert (
        by_strategy["neighborhood"]["retained_node_count"] == 5
    )

    assert by_strategy["forward"]["retention_ratio"] == 4 / 6
    assert by_strategy["backward"]["retention_ratio"] == 3 / 6
    assert (
        by_strategy["neighborhood"]["retention_ratio"] == 5 / 6
    )


def test_diagnostic_does_not_mutate_original_sample():
    sample = make_sample()

    original_node_ids = [
        node.node_id for node in sample.cfg["nodes"]
    ]
    original_edges = list(sample.cfg["edges"])

    run_pruning_diagnostics([sample])

    assert sample.pruned_cfg is None
    assert [
        node.node_id for node in sample.cfg["nodes"]
    ] == original_node_ids
    assert sample.cfg["edges"] == original_edges

# Add to src/tests/test_pruning_diagnostics.py

import csv

from src.debug.pruning_diagnostics import (
    run_pruning_diagnostics,
    write_diagnostics_csv,
)


def test_diagnostics_csv_export(tmp_path):
    rows = run_pruning_diagnostics([make_sample()])

    output_path = str(tmp_path / "pruning_diagnostics.csv")
    write_diagnostics_csv(rows, output_path)

    with open(output_path, "r") as csv_file:
        saved_rows = list(csv.DictReader(csv_file))

    assert len(saved_rows) == 3
    assert {row["strategy"] for row in saved_rows} == {
        "forward",
        "backward",
        "neighborhood",
    }


from types import SimpleNamespace


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


def test_all_strategies_use_same_original_graph():
    sample = make_chain_sample()

    original_node_ids = {
        node.node_id for node in sample.cfg["nodes"]
    }
    original_edges = list(sample.cfg["edges"])

    rows = run_pruning_diagnostics([sample], neighborhood_hops=2)

    assert len(rows) == 3

    by_strategy = {
        row["strategy"]: row for row in rows
    }

    assert set(by_strategy) == {
        "forward",
        "backward",
        "neighborhood",
    }

    # The seed is node 3 in a six-node chain.
    assert by_strategy["forward"]["retained_node_count"] == 4
    assert by_strategy["backward"]["retained_node_count"] == 3
    assert by_strategy["neighborhood"]["retained_node_count"] == 5

    for row in rows:
        assert row["original_node_count"] == 6
        assert row["seed_count"] == 1
        assert row["valid_seed_count"] == 1

    # Diagnostic runs must not mutate the original sample.
    assert {
        node.node_id for node in sample.cfg["nodes"]
    } == original_node_ids
    assert sample.cfg["edges"] == original_edges
    assert not hasattr(sample, "pruned_cfg")
