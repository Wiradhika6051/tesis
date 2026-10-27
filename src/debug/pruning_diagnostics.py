# src/debug/pruning_diagnostics.py

import copy
import csv
import os

from src.pruners.forward_slice_pruner import ForwardSlicePruner
from src.pruners.backward_slice_pruner import BackwardSlicePruner
from src.pruners.neighborhood_pruner import NeighborhoodPruner


def _get_sample_id(sample):
    """Build a readable identifier without requiring one specific schema."""
    parts = [
        getattr(sample, "repo", None),
        getattr(sample, "parent_commit", None),
        getattr(sample, "file_path", None),
        getattr(sample, "function_start", None),
        getattr(sample, "function_end", None),
    ]

    available = [str(part) for part in parts if part is not None]
    return "::".join(available) or str(
        getattr(sample, "sample_id", "unknown")
    )


def _unique_ids(values):
    """Preserve order while removing duplicate IDs."""
    return list(dict.fromkeys(values or []))


def _measure_sample(sample, strategy_name, pruner):
    # Each strategy gets its own copy of the original input.
    working_sample = copy.deepcopy(sample)

    original_nodes = set(working_sample.function_nodes or [])
    seed_nodes = _unique_ids(working_sample.seed_nodes)

    valid_seeds = [
        seed for seed in seed_nodes if seed in original_nodes
    ]

    original_node_count = len(original_nodes)

    result = pruner.prune(working_sample)
    pruned_cfg = getattr(result, "pruned_cfg", None)

    if pruned_cfg is None:
        raise ValueError(
            "{} returned no pruned_cfg for sample {}".format(
                strategy_name, _get_sample_id(sample)
            )
        )

    retained_node_ids = {
        node.node_id
        for node in pruned_cfg.get("nodes", [])
    }

    retained_node_count = len(retained_node_ids)

    retention_ratio = (
        float(retained_node_count) / original_node_count
        if original_node_count
        else 0.0
    )

    return {
        "sample_id": _get_sample_id(sample),
        "strategy": strategy_name,
        "original_node_count": original_node_count,
        "seed_count": len(seed_nodes),
        "valid_seed_count": len(valid_seeds),
        "retained_node_count": retained_node_count,
        "removed_node_count": (
            original_node_count - retained_node_count
        ),
        "retention_ratio": retention_ratio,
        "original_edge_count": len(
            sample.cfg.get("edges", [])
        ),
        "retained_edge_count": len(
            pruned_cfg.get("edges", [])
        ),
        "status": (
            "ok" if original_node_count else "empty_original_graph"
        ),
    }


def run_pruning_diagnostics(samples, neighborhood_hops=2):
    """
    Run all three pruning strategies against the same sample inputs.

    Returns one diagnostic row per sample per strategy.
    """
    strategies = [
        ("forward", ForwardSlicePruner()),
        ("backward", BackwardSlicePruner()),
        (
            "neighborhood",
            NeighborhoodPruner(hops=neighborhood_hops),
        ),
    ]

    rows = []

    for sample in samples:
        for strategy_name, pruner in strategies:
            rows.append(
                _measure_sample(
                    sample=sample,
                    strategy_name=strategy_name,
                    pruner=pruner,
                )
            )

    return rows


def write_diagnostics_csv(rows, output_path):
    """Write per-sample, per-strategy diagnostics to CSV."""
    if not rows:
        raise ValueError("No diagnostic rows to write.")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def print_diagnostics_summary(rows):
    """Print mean retention by strategy."""
    strategies = sorted({
        row["strategy"] for row in rows
    })

    print("\nPruning diagnostic summary")
    print("-" * 72)

    for strategy in strategies:
        strategy_rows = [
            row for row in rows
            if row["strategy"] == strategy
        ]

        if not strategy_rows:
            continue

        total_original = sum(
            row["original_node_count"]
            for row in strategy_rows
        )
        total_retained = sum(
            row["retained_node_count"]
            for row in strategy_rows
        )

        # Aggregate ratio is node-weighted, not the mean of
        # individual sample ratios.
        aggregate_ratio = (
            float(total_retained) / total_original
            if total_original
            else 0.0
        )

        print(
            "{}: samples={}, original_nodes={}, "
            "retained_nodes={}, aggregate_retention={:.2%}".format(
                strategy,
                len(strategy_rows),
                total_original,
                total_retained,
                aggregate_ratio,
            )
        )


import copy
import csv
import os

from src.pruners.forward_slice_pruner import ForwardSlicePruner
from src.pruners.backward_slice_pruner import BackwardSlicePruner
from src.pruners.neighborhood_pruner import NeighborhoodPruner


def _sample_id(sample):
    parts = [
        getattr(sample, "repo", None),
        getattr(sample, "parent_commit", None),
        getattr(sample, "file_path", None),
        getattr(sample, "function_start", None),
        getattr(sample, "function_end", None),
    ]
    values = [str(value) for value in parts if value is not None]
    return "::".join(values) or "unknown"


def _measure(sample, strategy, pruner):
    working = copy.deepcopy(sample)

    function_nodes = set(working.function_nodes or [])
    seed_nodes = set(working.seed_nodes or [])
    valid_seeds = seed_nodes & function_nodes

    original_node_count = len(function_nodes)

    # Record the original graph's size before pruning.
    original_edge_count = len(
        (working.cfg or {}).get("edges", [])
    )

    result = pruner.prune(working)
    pruned_cfg = result.pruned_cfg

    if pruned_cfg is None:
        raise ValueError(
            "{} returned no pruned_cfg for {}".format(
                strategy, _sample_id(sample)
            )
        )

    # The pruning utility retains original node IDs.
    retained_ids = {
        node.node_id
        for node in pruned_cfg.get("nodes", [])
    }

    # Count only AST nodes belonging to this function.
    retained_function_nodes = retained_ids & function_nodes
    retained_count = len(retained_function_nodes)

    retention_ratio = (
        float(retained_count) / original_node_count
        if original_node_count else 0.0
    )

    return {
        "sample_id": _sample_id(sample),
        "strategy": strategy,
        "original_node_count": original_node_count,
        "seed_count": len(seed_nodes),
        "valid_seed_count": len(valid_seeds),
        "invalid_seed_count": len(seed_nodes - function_nodes),
        "retained_node_count": retained_count,
        "removed_node_count": (
            original_node_count - retained_count
        ),
        "retention_ratio": retention_ratio,
        "original_edge_count": original_edge_count,
        "retained_edge_count": len(
            pruned_cfg.get("edges", [])
        ),
    }


def run_pruning_diagnostics(samples, neighborhood_hops=2):
    strategies = [
        ("forward", ForwardSlicePruner()),
        ("backward", BackwardSlicePruner()),
        (
            "neighborhood",
            NeighborhoodPruner(hops=neighborhood_hops),
        ),
    ]

    rows = []

    for sample in samples:
        for name, pruner in strategies:
            rows.append(_measure(sample, name, pruner))

    return rows


def write_diagnostics_csv(rows, output_path):
    if not rows:
        raise ValueError("No diagnostic results to write.")

    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def print_diagnostics_summary(rows):
    print("\nPruning diagnostic summary")
    print("=" * 72)

    for strategy in ("forward", "backward", "neighborhood"):
        subset = [
            row for row in rows
            if row["strategy"] == strategy
        ]

        if not subset:
            continue

        total_original = sum(
            row["original_node_count"] for row in subset
        )
        total_retained = sum(
            row["retained_node_count"] for row in subset
        )

        ratio = (
            float(total_retained) / total_original
            if total_original else 0.0
        )

        print(
            "{}: samples={}, original_nodes={}, "
            "retained_nodes={}, aggregate_retention={:.2%}".format(
                strategy,
                len(subset),
                total_original,
                total_retained,
                ratio,
            )
        )
