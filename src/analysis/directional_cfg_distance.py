from collections import defaultdict
from statistics import mean, median

from src.analysis.paired_slice_similarity import (
    get_slice_signatures
)


# ============================================================
# Helpers
# ============================================================

def _build_sample_lookup(samples):
    """
    Build:

        sample_id -> sample

    lookup for the paired test samples.
    """

    return {
        _get_sample_id(sample): sample
        for sample in samples
    }


def _get_sample_id(sample):
    """
    Construct the same identifier used by
    directional_results.
    """

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label
    )


def _node_line_map(sample):
    """
    Map normalized node signatures to source lines.

    Node IDs are intentionally not used because the
    directional comparison is based on semantic node
    signatures.
    """

    if sample.cfg is None:
        return {}

    result = defaultdict(list)

    for node in sample.cfg["nodes"]:

        signature = (
            node.node_type,
            _normalize_text(node.text)
        )

        if node.lineno is not None:

            result[signature].append(
                node.lineno
            )

    return result


def _normalize_text(text):

    if text is None:
        return ""

    return " ".join(
        text.strip().split()
    )


def _get_context_signatures(
    sample,
    forward
):
    """
    Get semantic signatures for one directional slice.
    """

    return get_slice_signatures(
        sample,
        forward=forward
    )


def _get_context_distances(
    sample,
    directional_result,
    forward
):
    """
    Calculate CFG/source-line distance of nodes that
    are unique to one directional slice.

    Distance is measured relative to the seed lines.

    Returns a list of distances.
    """

    if sample.cfg is None:
        return []

    if not sample.seed_nodes:
        return []

    if not sample.function_nodes:
        return []

    forward_signatures = _get_context_signatures(
        sample,
        forward=True
    )

    backward_signatures = _get_context_signatures(
        sample,
        forward=False
    )

    if forward:

        directional_signatures = (
            forward_signatures
            - backward_signatures
        )

    else:

        directional_signatures = (
            backward_signatures
            - forward_signatures
        )

    if not directional_signatures:
        return []

    line_map = _node_line_map(
        sample
    )

    seed_lines = []

    for node in sample.cfg["nodes"]:

        if node.node_id in sample.seed_nodes:

            if node.lineno is not None:

                seed_lines.append(
                    node.lineno
                )

    if not seed_lines:
        return []

    distances = []

    for signature in directional_signatures:

        lines = line_map.get(
            signature,
            []
        )

        if not lines:
            continue

        for line in lines:

            distance = min(
                abs(line - seed_line)
                for seed_line in seed_lines
            )

            distances.append(
                float(distance)
            )

    return distances


# ============================================================
# Distance buckets
# ============================================================

def _bucket_distance(distance):

    if distance <= 1:
        return "1"

    if distance <= 3:
        return "2-3"

    if distance <= 5:
        return "4-5"

    return ">5"


def _summarize_distances(distances):

    if not distances:

        return {
            "average": 0.0,
            "median": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "count": 0,
            "buckets": {
                "1": 0,
                "2-3": 0,
                "4-5": 0,
                ">5": 0
            }
        }

    buckets = {
        "1": 0,
        "2-3": 0,
        "4-5": 0,
        ">5": 0
    }

    for distance in distances:

        bucket = _bucket_distance(
            distance
        )

        buckets[bucket] += 1

    return {
        "average": mean(distances),
        "median": median(distances),
        "minimum": min(distances),
        "maximum": max(distances),
        "count": len(distances),
        "buckets": buckets
    }


# ============================================================
# Main analysis
# ============================================================

def analyze_directional_cfg_distance(
    directional_results,
    forward_samples,
    backward_samples
):
    """
    Analyze CFG/source-line distance of directional-only
    context.

    The analysis is performed separately for:

        BOTH_CORRECT
        BOTH_WRONG
        FORWARD_CORRECT
        BACKWARD_CORRECT

    For each outcome we measure:

        forward-only distance
        backward-only distance

    Distance is measured from the nearest seed node.
    """

    forward_lookup = _build_sample_lookup(
        forward_samples
    )

    backward_lookup = _build_sample_lookup(
        backward_samples
    )

    grouped = defaultdict(list)

    for result in directional_results:

        outcome = result.get(
            "outcome"
        )

        if outcome not in (
            "BOTH_CORRECT",
            "BOTH_WRONG",
            "FORWARD_CORRECT",
            "BACKWARD_CORRECT"
        ):

            continue

        sample_id = result.get(
            "sample_id"
        )

        forward_sample = (
            forward_lookup.get(
                sample_id
            )
        )

        backward_sample = (
            backward_lookup.get(
                sample_id
            )
        )

        if (
            forward_sample is None
            or
            backward_sample is None
        ):

            continue

        forward_distances = (
            _get_context_distances(
                forward_sample,
                result,
                forward=True
            )
        )

        backward_distances = (
            _get_context_distances(
                backward_sample,
                result,
                forward=False
            )
        )

        grouped[outcome].append({

            "sample_id": sample_id,

            "forward_distances":
                forward_distances,

            "backward_distances":
                backward_distances,

            "forward":
                _summarize_distances(
                    forward_distances
                ),

            "backward":
                _summarize_distances(
                    backward_distances
                )

        })

    return dict(
        grouped
    )


# ============================================================
# Printing
# ============================================================

def _print_distance_distribution(
    name,
    summary
):

    print()
    print(
        f"{name} distance distribution:"
    )

    print(
        f"  Average : "
        f"{summary['average']:.2f}"
    )

    print(
        f"  Median  : "
        f"{summary['median']:.2f}"
    )

    print(
        f"  Minimum : "
        f"{summary['minimum']:.2f}"
    )

    print(
        f"  Maximum : "
        f"{summary['maximum']:.2f}"
    )

    print()
    print("  Buckets:")

    for bucket in (
        "1",
        "2-3",
        "4-5",
        ">5"
    ):

        print(
            f"    {bucket:<4}: "
            f"{summary['buckets'][bucket]}"
        )


def print_directional_cfg_distance(
    results
):

    print()
    print("=" * 100)
    print(
        "CFG DISTANCE BY PREDICTION OUTCOME"
    )
    print("=" * 100)

    outcome_order = [

        "BOTH_CORRECT",
        "BOTH_WRONG",
        "FORWARD_CORRECT",
        "BACKWARD_CORRECT"

    ]

    for outcome in outcome_order:

        rows = results.get(
            outcome,
            []
        )

        print()
        print(
            outcome
        )

        print(
            "-" * 60
        )

        print(
            f"Samples: {len(rows)}"
        )

        if not rows:
            continue

        forward_distances = []

        backward_distances = []

        for row in rows:

            forward_distances.extend(
                row["forward_distances"]
            )

            backward_distances.extend(
                row["backward_distances"]
            )

        forward_summary = (
            _summarize_distances(
                forward_distances
            )
        )

        backward_summary = (
            _summarize_distances(
                backward_distances
            )
        )

        _print_distance_distribution(
            "Forward-only",
            forward_summary
        )

        _print_distance_distribution(
            "Backward-only",
            backward_summary
        )

        print()
        print(
            "Normalized directional context"
        )

        print(
            "-" * 60
        )

        total_forward = (
            forward_summary["count"]
        )

        total_backward = (
            backward_summary["count"]
        )

        if total_forward > 0:

            forward_local = (

                forward_summary["buckets"]["1"]
                +
                forward_summary["buckets"]["2-3"]

            ) / total_forward

            forward_distant = (

                forward_summary["buckets"][">5"]

            ) / total_forward

        else:

            forward_local = 0.0
            forward_distant = 0.0

        if total_backward > 0:

            backward_local = (

                backward_summary["buckets"]["1"]
                +
                backward_summary["buckets"]["2-3"]

            ) / total_backward

            backward_distant = (

                backward_summary["buckets"][">5"]

            ) / total_backward

        else:

            backward_local = 0.0
            backward_distant = 0.0

        print(
            f"Forward local (1-3): "
            f"{forward_local:.2%}"
        )

        print(
            f"Forward distant (>5): "
            f"{forward_distant:.2%}"
        )

        print(
            f"Backward local (1-3): "
            f"{backward_local:.2%}"
        )

        print(
            f"Backward distant (>5): "
            f"{backward_distant:.2%}"
        )


# ============================================================
# Direct comparison of winning directions
# ============================================================

def print_winner_cfg_distance(
    results
):

    print()
    print("=" * 100)
    print(
        "FORWARD-CORRECT VS BACKWARD-CORRECT CFG DISTANCE"
    )
    print("=" * 100)

    forward_rows = results.get(
        "FORWARD_CORRECT",
        []
    )

    backward_rows = results.get(
        "BACKWARD_CORRECT",
        []
    )

    def average(rows, direction):

        values = []

        for row in rows:

            values.extend(
                row[
                    f"{direction}_distances"
                ]
            )

        if not values:
            return 0.0

        return mean(values)

    print()
    print(
        f"{'Metric':<40}"
        f"{'Forward correct':>20}"
        f"{'Backward correct':>20}"
    )

    print("-" * 80)

    print(
        f"{'Forward-only average distance':<40}"
        f"{average(forward_rows, 'forward'):>20.2f}"
        f"{average(backward_rows, 'forward'):>20.2f}"
    )

    print(
        f"{'Backward-only average distance':<40}"
        f"{average(forward_rows, 'backward'):>20.2f}"
        f"{average(backward_rows, 'backward'):>20.2f}"
    )