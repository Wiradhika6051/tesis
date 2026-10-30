
import copy
import csv
import os

from src.pruners.forward_slice_pruner import ForwardSlicePruner
from src.pruners.backward_slice_pruner import BackwardSlicePruner
from src.pruners.neighborhood_pruner import NeighborhoodPruner


DEFAULT_NEIGHBORHOOD_RETENTION_CUTOFF = 0.80
DEFAULT_IDENTICAL_JACCARD_CUTOFF = 0.95


def get_sample_id(sample):
    """Build a readable identifier for a sample."""
    parts = [
        getattr(sample, "repo", None),
        getattr(sample, "parent_commit", None),
        getattr(sample, "file_path", None),
        getattr(sample, "function_start", None),
        getattr(sample, "function_end", None),
    ]

    values = [str(value) for value in parts if value is not None]
    return "::".join(values) or "unknown"


def jaccard_similarity(set_a, set_b):
    """Calculate Jaccard similarity between two node-ID sets."""
    union = set_a | set_b

    if not union:
        return 0.0

    return float(len(set_a & set_b)) / len(union)


def run_strategies(sample, neighborhood_hops=2):
    """
    Run every strategy on an independent copy of the same sample.

    Returns:
        original_nodes: valid function-scoped CFG node IDs
        seed_nodes: valid seed IDs within the function
        retained: strategy name -> retained original node IDs
    """
    cfg = sample.cfg or {}

    cfg_node_ids = {
        node.node_id
        for node in cfg.get("nodes", [])
    }

    function_nodes = set(sample.function_nodes or [])

    # Denominator: nodes that both exist in the CFG and belong
    # to the function being analyzed.
    original_nodes = function_nodes & cfg_node_ids

    seed_nodes = (
        set(sample.seed_nodes or []) & original_nodes
    )

    strategies = [
        ("forward", ForwardSlicePruner()),
        ("backward", BackwardSlicePruner()),
        (
            "neighborhood",
            NeighborhoodPruner(hops=neighborhood_hops),
        ),
    ]

    retained = {}

    for strategy_name, pruner in strategies:
        working = copy.deepcopy(sample)
        result = pruner.prune(working)
        pruned_cfg = getattr(result, "pruned_cfg", None)

        if pruned_cfg is None:
            raise ValueError(
                "{} returned no pruned_cfg for {}".format(
                    strategy_name, get_sample_id(sample)
                )
            )

        retained_ids = {
            node.node_id
            for node in pruned_cfg.get("nodes", [])
        }

        # Only count valid original function nodes.
        retained[strategy_name] = retained_ids & original_nodes

    return original_nodes, seed_nodes, retained


def analyze_sample(
    sample,
    neighborhood_hops=2,
    neighborhood_retention_cutoff=(
        DEFAULT_NEIGHBORHOOD_RETENTION_CUTOFF
    ),
    identical_jaccard_cutoff=(
        DEFAULT_IDENTICAL_JACCARD_CUTOFF
    ),
):
    """Calculate retention and pairwise overlap metrics."""
    original_nodes, seed_nodes, retained = run_strategies(
        sample,
        neighborhood_hops=neighborhood_hops,
    )

    original_count = len(original_nodes)

    retention = {
        name: (
            float(len(nodes)) / original_count
            if original_count else 0.0
        )
        for name, nodes in retained.items()
    }

    forward = retained["forward"]
    backward = retained["backward"]
    neighborhood = retained["neighborhood"]

    forward_backward_intersection = forward & backward
    forward_backward_union = forward | backward

    row = {
        "sample_id": get_sample_id(sample),
        "original_node_count": original_count,
        "seed_count": len(seed_nodes),
        "forward_retained_count": len(forward),
        "backward_retained_count": len(backward),
        "neighborhood_retained_count": len(neighborhood),
        "forward_retention_ratio": retention["forward"],
        "backward_retention_ratio": retention["backward"],
        "neighborhood_retention_ratio": retention["neighborhood"],
        "forward_backward_intersection_count": len(
            forward_backward_intersection
        ),
        "forward_backward_union_count": len(
            forward_backward_union
        ),
        "forward_backward_jaccard": jaccard_similarity(
            forward, backward
        ),
        "neighborhood_over_80_percent": (
            retention["neighborhood"]
            > neighborhood_retention_cutoff
        ),
    }

    pairwise_rows = []

    pairs = [
        ("forward", "backward"),
        ("forward", "neighborhood"),
        ("backward", "neighborhood"),
    ]

    for strategy_a, strategy_b in pairs:
        nodes_a = retained[strategy_a]
        nodes_b = retained[strategy_b]

        intersection = nodes_a & nodes_b
        union = nodes_a | nodes_b
        similarity = jaccard_similarity(nodes_a, nodes_b)

        pairwise_rows.append({
            "sample_id": get_sample_id(sample),
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "intersection_count": len(intersection),
            "union_count": len(union),
            "jaccard_similarity": similarity,
            "identical_node_sets": nodes_a == nodes_b,
            "nearly_identical": (
                similarity >= identical_jaccard_cutoff
                and bool(union)
            ),
        })

    return row, pairwise_rows


def run_overlap_diagnostics(
    samples,
    neighborhood_hops=2,
    neighborhood_retention_cutoff=(
        DEFAULT_NEIGHBORHOOD_RETENTION_CUTOFF
    ),
    identical_jaccard_cutoff=(
        DEFAULT_IDENTICAL_JACCARD_CUTOFF
    ),
):
    """Analyze all samples and return summary and pairwise rows."""
    summary_rows = []
    pairwise_rows = []

    for sample in samples:
        summary, pairs = analyze_sample(
            sample,
            neighborhood_hops=neighborhood_hops,
            neighborhood_retention_cutoff=(
                neighborhood_retention_cutoff
            ),
            identical_jaccard_cutoff=identical_jaccard_cutoff,
        )

        summary_rows.append(summary)
        pairwise_rows.extend(pairs)

    return summary_rows, pairwise_rows


def write_csv(rows, output_path):
    """Write a list of dictionaries to CSV."""
    if not rows:
        raise ValueError(
            "No rows to write: {}".format(output_path)
        )

    output_dir = os.path.dirname(output_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def print_overlap_summary(summary_rows, pairwise_rows):
    """Print high-level diagnostic counts."""
    print("\nPruning overlap diagnostic")
    print("=" * 72)
    print("Samples analyzed: {}".format(len(summary_rows)))

    high_retention = [
        row for row in summary_rows
        if row["neighborhood_over_80_percent"]
    ]

    print(
        "Neighborhood retention > 80%: {} / {}".format(
            len(high_retention), len(summary_rows)
        )
    )

    forward_backward_high_overlap = [
        row for row in summary_rows
        if row["forward_backward_jaccard"] >= 0.95
        and row["forward_backward_union_count"] > 0
    ]

    print(
        "Forward/backward Jaccard >= 0.95: {} / {}".format(
            len(forward_backward_high_overlap),
            len(summary_rows),
        )
    )

    identical = [
        row for row in pairwise_rows
        if row["identical_node_sets"]
    ]

    nearly_identical = [
        row for row in pairwise_rows
        if row["nearly_identical"]
    ]

    print(
        "Identical strategy pairs: {}".format(len(identical))
    )
    print(
        "Nearly identical strategy pairs (Jaccard >= 0.95): "
        "{}".format(len(nearly_identical))
    )
