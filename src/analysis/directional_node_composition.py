from collections import Counter, defaultdict


# ============================================================
# Helpers
# ============================================================

def _sample_id(sample):
    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label
    )


def _normalize_text(text):

    if text is None:
        return ""

    return " ".join(
        text.strip().split()
    )


def _node_signature(node):

    return (
        node.node_type,
        _normalize_text(node.text)
    )


def _slice_signatures(
    sample,
    forward
):
    """
    Get semantic node signatures belonging to one
    directional slice.
    """

    if sample.cfg is None:
        return set()

    if not sample.seed_nodes:
        return set()

    if not sample.function_nodes:
        return set()

    # Import the actual slice calculation used by
    # your existing directional analysis.
    from src.analysis.slice_analysis import (
        get_slice_nodes
    )

    node_ids = get_slice_nodes(

        sample.cfg,

        set(sample.seed_nodes),

        set(sample.function_nodes),

        forward=forward

    )

    node_lookup = {

        node.node_id: node
        for node in sample.cfg["nodes"]

    }

    return {

        _node_signature(
            node_lookup[node_id]
        )

        for node_id in node_ids

        if node_id in node_lookup

    }


def _get_directional_types(
    forward_sample,
    backward_sample
):
    """
    Return node-type counters for:

        forward-only
        backward-only
    """

    forward_signatures = _slice_signatures(
        forward_sample,
        forward=True
    )

    backward_signatures = _slice_signatures(
        backward_sample,
        forward=False
    )

    forward_only = (
        forward_signatures
        - backward_signatures
    )

    backward_only = (
        backward_signatures
        - forward_signatures
    )

    forward_types = Counter(
        node_type
        for node_type, _ in forward_only
    )

    backward_types = Counter(
        node_type
        for node_type, _ in backward_only
    )

    return (
        forward_types,
        backward_types
    )


# ============================================================
# Build composition results
# ============================================================

def analyze_directional_node_composition(
    directional_results,
    forward_samples,
    backward_samples
):
    """
    Build directional node-type composition grouped
    by prediction outcome.
    """

    forward_lookup = {
        _sample_id(sample): sample
        for sample in forward_samples
    }

    backward_lookup = {
        _sample_id(sample): sample
        for sample in backward_samples
    }

    grouped = defaultdict(
        lambda: {
            "samples": 0,
            "forward_types": Counter(),
            "backward_types": Counter(),
            "records": []
        }
    )

    seen = set()

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

        # Protect against duplicate directional results.
        if sample_id in seen:
            continue

        forward_sample = forward_lookup.get(
            sample_id
        )

        backward_sample = backward_lookup.get(
            sample_id
        )

        if (
            forward_sample is None
            or
            backward_sample is None
        ):
            continue

        (
            forward_types,
            backward_types
        ) = _get_directional_types(
            forward_sample,
            backward_sample
        )

        grouped[outcome]["samples"] += 1

        grouped[outcome]["forward_types"].update(
            forward_types
        )

        grouped[outcome]["backward_types"].update(
            backward_types
        )

        grouped[outcome]["records"].append({

            "sample_id": sample_id,

            "forward_types":
                forward_types,

            "backward_types":
                backward_types

        })

        seen.add(
            sample_id
        )

    return dict(
        grouped
    )


# ============================================================
# Printing
# ============================================================

def _print_counter(
    counter
):
    """
    Print node types ordered by frequency.
    """

    total = sum(
        counter.values()
    )

    if total == 0:

        print(
            "None"
        )

        return

    for node_type, count in counter.most_common():

        percentage = (
            count / total * 100
        )

        print(
            f"{node_type:<30}"
            f"{count:>6}"
            f"{percentage:>10.2f}%"
        )


def print_directional_node_composition(
    directional_results,
    forward_samples=None,
    backward_samples=None
):
    """
    Print directional node-type composition.

    The function accepts paired samples because the
    current directional_results schema stores counts,
    not node-type information.
    """

    if (
        forward_samples is None
        or
        backward_samples is None
    ):

        raise ValueError(
            "forward_samples and backward_samples "
            "are required for node composition analysis."
        )

    results = analyze_directional_node_composition(

        directional_results,

        forward_samples,

        backward_samples

    )

    outcome_order = [

        "BOTH_CORRECT",
        "BOTH_WRONG",
        "FORWARD_CORRECT",
        "BACKWARD_CORRECT"

    ]

    for outcome in outcome_order:

        group = results.get(
            outcome
        )

        if group is None:
            continue

        print()
        print(
            "=" * 80
        )

        print(
            "DIRECTIONAL NODE COMPOSITION"
        )

        print(
            f"Outcome: {outcome}"
        )

        print(
            "=" * 80
        )

        print(
            f"Samples: "
            f"{group['samples']}"
        )

        print()
        print(
            "FORWARD-ONLY NODE TYPES"
        )

        print(
            "-" * 60
        )

        _print_counter(
            group["forward_types"]
        )

        print()
        print(
            "BACKWARD-ONLY NODE TYPES"
        )

        print(
            "-" * 60
        )

        _print_counter(
            group["backward_types"]
        )