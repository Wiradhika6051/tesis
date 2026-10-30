
import copy
import csv
import os

from src.pruners.neighborhood_pruner import NeighborhoodPruner
from src.pruners.cfg_traversal import traverse_cfg


DEFAULT_RETENTION_CUTOFF = 0.80


def sample_identifier(sample):
    parts = [
        getattr(sample, "repo", None),
        getattr(sample, "parent_commit", None),
        getattr(sample, "file_path", None),
        getattr(sample, "function_start", None),
        getattr(sample, "function_end", None),
    ]
    parts = [str(value) for value in parts if value is not None]
    return "::".join(parts) or "unknown"


def inspect_sample_structure(sample):
    """Collect structural metrics without modifying the sample."""
    cfg = sample.cfg or {}
    nodes = cfg.get("nodes", [])
    edges = cfg.get("edges", [])

    node_by_id = {
        node.node_id: node
        for node in nodes
    }
    cfg_ids = set(node_by_id)
    function_ids = set(sample.function_nodes or [])
    seed_ids = set(sample.seed_nodes or [])

    valid_function_ids = function_ids & cfg_ids
    valid_seeds = seed_ids & valid_function_ids

    virtual_ids = {
        node_id
        for node_id, node in node_by_id.items()
        if getattr(node, "lineno", None) is not None
        and node.lineno < 1
    }

    ordinary_ids = cfg_ids - virtual_ids
    outside_function_ids = cfg_ids - function_ids

    valid_edges = [
        (src, dst)
        for edge in edges
        if isinstance(edge, (tuple, list)) and len(edge) == 2
        for src, dst in [edge]
        if src in cfg_ids and dst in cfg_ids
    ]

    edges_touching_outside_nodes = [
        (src, dst)
        for src, dst in valid_edges
        if src not in function_ids or dst not in function_ids
    ]

    internal_function_edges = [
        (src, dst)
        for src, dst in valid_edges
        if src in function_ids and dst in function_ids
    ]

    # Virtual nodes can be traversed without consuming AST hops.
    # Count virtual-to-virtual edges as a structural warning signal.
    virtual_to_virtual_edges = [
        (src, dst)
        for src, dst in valid_edges
        if src in virtual_ids and dst in virtual_ids
    ]

    return {
        "cfg_ids": cfg_ids,
        "function_ids": function_ids,
        "valid_function_ids": valid_function_ids,
        "seed_ids": seed_ids,
        "valid_seeds": valid_seeds,
        "virtual_ids": virtual_ids,
        "ordinary_ids": ordinary_ids,
        "outside_function_ids": outside_function_ids,
        "valid_edges": valid_edges,
        "internal_function_edges": internal_function_edges,
        "edges_touching_outside_nodes": edges_touching_outside_nodes,
        "virtual_to_virtual_edges": virtual_to_virtual_edges,
    }


def analyze_hop_budget(sample, hops):
    """Run neighborhood traversal without mutating the original sample."""
    working = copy.deepcopy(sample)

    reachable = traverse_cfg(
        cfg=working.cfg,
        seed_nodes=working.seed_nodes,
        function_nodes=working.function_nodes,
        direction="undirected",
        max_ast_hops=hops,
    )

    function_ids = set(working.function_nodes or [])
    cfg_ids = {
        node.node_id for node in working.cfg.get("nodes", [])
    }
    valid_function_ids = function_ids & cfg_ids

    return {
        "hops": "unlimited" if hops is None else hops,
        "reachable_count": len(reachable),
        "valid_function_node_count": len(valid_function_ids),
        "retention_ratio": (
            float(len(reachable)) / len(valid_function_ids)
            if valid_function_ids else 0.0
        ),
        "reachable_ids": reachable,
    }


def investigate_sample(
    sample,
    retention_cutoff=DEFAULT_RETENTION_CUTOFF,
    hop_budgets=(0, 1, 2, 3, None),
):
    structure = inspect_sample_structure(sample)

    function_ids = structure["valid_function_ids"]
    seed_ids = structure["seed_ids"]
    valid_seeds = structure["valid_seeds"]
    virtual_ids = structure["virtual_ids"]
    ordinary_ids = structure["ordinary_ids"]

    function_count = len(function_ids)
    seed_count = len(seed_ids)
    valid_seed_count = len(valid_seeds)

    seed_ratio = (
        float(valid_seed_count) / function_count
        if function_count else 0.0
    )

    # Sensitivity analysis using the same original graph and seeds.
    hop_results = [
        analyze_hop_budget(sample, hops)
        for hops in hop_budgets
    ]

    hop_2 = next(
        (
            result for result in hop_results
            if result["hops"] == 2
        ),
        None,
    )

    high_retention = (
        hop_2 is not None
        and hop_2["retention_ratio"] > retention_cutoff
    )

    # Heuristic flags are prompts for inspection, not diagnoses.
    flags = []

    if function_count and seed_ratio > 0.25:
        flags.append("high_seed_ratio")

    if seed_ids - function_ids:
        flags.append("seeds_outside_function_scope")

    if structure["outside_function_ids"]:
        flags.append("cfg_contains_nodes_outside_function_scope")

    if structure["edges_touching_outside_nodes"]:
        flags.append("edges_cross_function_scope_boundary")

    if virtual_ids:
        flags.append("contains_virtual_nodes")

    if structure["virtual_to_virtual_edges"]:
        flags.append("contains_virtual_to_virtual_edges")

    if high_retention:
        flags.append("neighborhood_retention_over_cutoff")

    row = {
        "sample_id": sample_identifier(sample),
        "cfg_node_count": len(structure["cfg_ids"]),
        "function_node_count": function_count,
        "seed_count": seed_count,
        "valid_seed_count": valid_seed_count,
        "invalid_seed_count": len(seed_ids - function_ids),
        "seed_ratio": seed_ratio,
        "virtual_node_count": len(virtual_ids),
        "ordinary_cfg_node_count": len(ordinary_ids),
        "function_internal_edge_count": len(
            structure["internal_function_edges"]
        ),
        "edges_touching_outside_function_count": len(
            structure["edges_touching_outside_nodes"]
        ),
        "virtual_to_virtual_edge_count": len(
            structure["virtual_to_virtual_edges"]
        ),
        "neighborhood_hop2_retained_count": (
            hop_2["reachable_count"] if hop_2 else None
        ),
        "neighborhood_hop2_retention_ratio": (
            hop_2["retention_ratio"] if hop_2 else None
        ),
        "high_retention": high_retention,
        "investigation_flags": ";".join(flags),
    }

    hop_rows = []

    for result in hop_results:
        hop_rows.append({
            "sample_id": sample_identifier(sample),
            "hops": result["hops"],
            "valid_function_node_count": (
                result["valid_function_node_count"]
            ),
            "reachable_count": result["reachable_count"],
            "retention_ratio": result["retention_ratio"],
        })

    return row, hop_rows


def run_investigation(
    samples,
    retention_cutoff=DEFAULT_RETENTION_CUTOFF,
    hop_budgets=(0, 1, 2, 3, None),
):
    summary_rows = []
    hop_rows = []

    for sample in samples:
        summary, sensitivity = investigate_sample(
            sample,
            retention_cutoff=retention_cutoff,
            hop_budgets=hop_budgets,
        )
        summary_rows.append(summary)
        hop_rows.extend(sensitivity)

    return summary_rows, hop_rows


def write_csv(rows, output_path):
    if not rows:
        raise ValueError(
            "No rows to write: {}".format(output_path)
        )

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


def print_investigation_summary(summary_rows):
    high_retention = [
        row for row in summary_rows
        if row["high_retention"]
    ]

    print("\nExcessive-retention investigation")
    print("=" * 72)
    print("Samples analyzed: {}".format(len(summary_rows)))
    print(
        "Samples over retention cutoff: {} / {}".format(
            len(high_retention), len(summary_rows)
        )
    )

    print("\nFlag counts (flags may overlap):")

    flag_counts = {}

    for row in summary_rows:
        for flag in row["investigation_flags"].split(";"):
            if flag:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

    for flag, count in sorted(
        flag_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print("  {}: {}".format(flag, count))

    print("\nRepresentative high-retention samples:")

    for row in high_retention[:10]:
        print(
            "  {} | retention={} | seeds={}/{} | flags={}".format(
                row["sample_id"],
                (
                    "{:.1%}".format(
                        row["neighborhood_hop2_retention_ratio"]
                    )
                    if row["neighborhood_hop2_retention_ratio"]
                    is not None else "N/A"
                ),
                row["valid_seed_count"],
                row["function_node_count"],
                row["investigation_flags"] or "none",
            )
        )
