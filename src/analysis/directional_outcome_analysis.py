from collections import defaultdict, Counter
import statistics


def _sample_id(sample):
    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label
    )


def _get_slice_nodes(
    sample,
    forward
):
    """
    Get the node IDs belonging to a forward/backward slice.

    Uses the existing pruner output stored in sample.pruned_cfg.
    """

    if sample.pruned_cfg is None:
        return set()

    return set(
        range(
            len(
                sample.pruned_cfg["nodes"]
            )
        )
    )


def _get_slice_signatures(
    sample,
    forward
):
    """
    Return signatures of nodes in the directional slice.

    IMPORTANT:
    This should use the same signature logic as the existing
    paired_slice_similarity analysis if available.
    """

    nodes = _get_slice_nodes(
        sample,
        forward
    )

    signatures = set()

    for node_id in nodes:

        node = sample.pruned_cfg["nodes"][node_id]

        signatures.add(
            (
                node.node_type,
                node.text.strip()
            )
        )

    return signatures


def _node_line_map(sample):

    result = {}

    if sample.cfg is None:
        return result

    for node_id, node in enumerate(
        sample.cfg["nodes"]
    ):

        result[node_id] = node.lineno

    return result


def _directional_slice_statistics(
    sample
):
    """
    Calculate structural characteristics of the
    forward/backward slices for one sample.
    """

    forward_nodes = get_slice_nodes(
        sample,
        forward=True
    )

    backward_nodes = get_slice_nodes(
        sample,
        forward=False
    )

    overlap = (
        forward_nodes
        &
        backward_nodes
    )

    forward_only = (
        forward_nodes
        -
        backward_nodes
    )

    backward_only = (
        backward_nodes
        -
        forward_nodes
    )

    seed_lines = set(
        sample.seed_lines
        or []
    )

    line_map = _node_line_map(
        sample
    )

    forward_only_before = 0
    forward_only_after = 0

    backward_only_before = 0
    backward_only_after = 0

    for node_id in forward_only:

        line = line_map.get(
            node_id
        )

        if line is None:
            continue

        if line < min(seed_lines, default=line):
            forward_only_before += 1

        elif line > max(seed_lines, default=line):
            forward_only_after += 1

    for node_id in backward_only:

        line = line_map.get(
            node_id
        )

        if line is None:
            continue

        if line < min(seed_lines, default=line):
            backward_only_before += 1

        elif line > max(seed_lines, default=line):
            backward_only_after += 1

    return {

        "forward_size":
            len(forward_nodes),

        "backward_size":
            len(backward_nodes),

        "overlap":
            len(overlap),

        "forward_only":
            len(forward_only),

        "backward_only":
            len(backward_only),

        "forward_only_before":
            forward_only_before,

        "forward_only_after":
            forward_only_after,

        "backward_only_before":
            backward_only_before,

        "backward_only_after":
            backward_only_after
    }


def get_slice_nodes(
    sample,
    forward
):
    """
    Obtain the actual directional slice from the sample.

    Replace this body with the same get_slice_nodes call
    used by your existing slice analysis if it is imported
    from another module.
    """

    from src.analysis.slice_analysis import get_slice_nodes

    return set(
        get_slice_nodes(
            sample.cfg,
            set(sample.seed_nodes),
            set(sample.function_nodes),
            forward=forward
        )
    )


def compare_directional_outcomes(
    forward_samples,
    backward_samples,
    forward_results,
    backward_results
):
    """
    Compare structural differences between forward and backward
    slices against model prediction outcomes.

    The forward and backward sample lists must represent the
    same test samples.
    """
    seen_sample_ids = set()
    forward_predictions = (
        forward_results[
            "predictions"
        ].tolist()
    )

    backward_predictions = (
        backward_results[
            "predictions"
        ].tolist()
    )

    forward_labels = (
        forward_results[
            "labels"
        ].tolist()
    )

    backward_labels = (
        backward_results[
            "labels"
        ].tolist()
    )

    #
    # Build prediction lookup by sample ID.
    #
    forward_lookup = {}
    backward_lookup = {}

    for sample, prediction, label in zip(
        forward_samples,
        forward_predictions,
        forward_labels
    ):

        sample_id = _sample_id(
            sample
        )

        forward_lookup[
            sample_id
        ] = {
            "prediction":
                prediction,
            "label":
                label
        }

    for sample, prediction, label in zip(
        backward_samples,
        backward_predictions,
        backward_labels
    ):

        sample_id = _sample_id(
            sample
        )

        backward_lookup[
            sample_id
        ] = {
            "prediction":
                prediction,
            "label":
                label
        }

    records = []

    for sample in forward_samples:

        sample_id = _sample_id(
            sample
        )

        forward = forward_lookup.get(
            sample_id
        )

        backward = backward_lookup.get(
            sample_id
        )

        if (
            forward is None
            or
            backward is None
        ):
            continue

        label = forward["label"]

        forward_prediction = (
            forward["prediction"]
        )

        backward_prediction = (
            backward["prediction"]
        )

        forward_correct = (
            forward_prediction
            ==
            label
        )

        backward_correct = (
            backward_prediction
            ==
            label
        )

        if (
            forward_correct
            and
            backward_correct
        ):

            outcome = "BOTH_CORRECT"

        elif (
            not forward_correct
            and
            not backward_correct
        ):

            outcome = "BOTH_WRONG"

        elif forward_correct:

            outcome = "FORWARD_CORRECT"

        else:

            outcome = "BACKWARD_CORRECT"

        stats = _directional_slice_statistics(
            sample
        )

        #
        # Directional asymmetry.
        #
        stats["size_difference"] = (
            stats["forward_size"]
            -
            stats["backward_size"]
        )

        stats["unique_difference"] = (
            stats["forward_only"]
            -
            stats["backward_only"]
        )

        stats["before_difference"] = (
            stats["backward_only_before"]
            -
            stats["forward_only_before"]
        )

        stats["after_difference"] = (
            stats["forward_only_after"]
            -
            stats["backward_only_after"]
        )

        stats.update({

            "sample_id":
                sample_id,

            "label":
                label,

            "forward_prediction":
                forward_prediction,

            "backward_prediction":
                backward_prediction,

            "outcome":
                outcome

        })

        records.append(
            stats
        )
        if sample_id in seen_sample_ids:
            print(
                "Skipping duplicate sample:",
                sample_id
            )
            continue

        seen_sample_ids.add(
            sample_id
        )

    return records


def _mean(
    records,
    key
):

    values = [
        r[key]
        for r in records
    ]

    if not values:
        return 0.0

    return statistics.mean(
        values
    )


def print_directional_outcome_analysis(
    records
):

    groups = defaultdict(
        list
    )

    for record in records:

        groups[
            record["outcome"]
        ].append(
            record
        )

    print()
    print("=" * 80)
    print(
        "SLICE CHARACTERISTICS BY PREDICTION OUTCOME"
    )
    print("=" * 80)

    outcomes = [

        "BOTH_CORRECT",
        "BOTH_WRONG",
        "FORWARD_CORRECT",
        "BACKWARD_CORRECT"

    ]

    for outcome in outcomes:

        group = groups.get(
            outcome,
            []
        )

        print()
        print(outcome)
        print("-" * 60)

        print(
            f"Samples: {len(group)}"
        )

        if not group:
            continue

        print(
            f"Average forward slice       : "
            f"{_mean(group, 'forward_size'):.2f}"
        )

        print(
            f"Average backward slice      : "
            f"{_mean(group, 'backward_size'):.2f}"
        )

        print(
            f"Average overlap             : "
            f"{_mean(group, 'overlap'):.2f}"
        )

        print(
            f"Average forward-only        : "
            f"{_mean(group, 'forward_only'):.2f}"
        )

        print(
            f"Average backward-only       : "
            f"{_mean(group, 'backward_only'):.2f}"
        )

        print(
            f"Average forward-only before : "
            f"{_mean(group, 'forward_only_before'):.2f}"
        )

        print(
            f"Average forward-only after  : "
            f"{_mean(group, 'forward_only_after'):.2f}"
        )

        print(
            f"Average backward-only before: "
            f"{_mean(group, 'backward_only_before'):.2f}"
        )

        print(
            f"Average backward-only after : "
            f"{_mean(group, 'backward_only_after'):.2f}"
        )

        print(
            f"Average size difference     : "
            f"{_mean(group, 'size_difference'):.2f}"
        )

        print(
            f"Average unique difference   : "
            f"{_mean(group, 'unique_difference'):.2f}"
        )

        print(
            f"Average before difference   : "
            f"{_mean(group, 'before_difference'):.2f}"
        )

        print(
            f"Average after difference    : "
            f"{_mean(group, 'after_difference'):.2f}"
        )

    #
    # --------------------------------------------------
    # Direct comparison of the two disagreement groups.
    # --------------------------------------------------
    #

    forward_correct = groups.get(
        "FORWARD_CORRECT",
        []
    )

    backward_correct = groups.get(
        "BACKWARD_CORRECT",
        []
    )

    print()
    print("=" * 80)
    print(
        "FORWARD-CORRECT VS BACKWARD-CORRECT"
    )
    print("=" * 80)

    metrics = [

        (
            "Forward slice",
            "forward_size"
        ),

        (
            "Backward slice",
            "backward_size"
        ),

        (
            "Overlap",
            "overlap"
        ),

        (
            "Forward-only",
            "forward_only"
        ),

        (
            "Backward-only",
            "backward_only"
        ),

        (
            "Forward-only before",
            "forward_only_before"
        ),

        (
            "Forward-only after",
            "forward_only_after"
        ),

        (
            "Backward-only before",
            "backward_only_before"
        ),

        (
            "Backward-only after",
            "backward_only_after"
        )

    ]

    print()

    print(
        f"{'Metric':<30}"
        f"{'Forward correct':>20}"
        f"{'Backward correct':>20}"
    )

    print("-" * 70)

    for name, key in metrics:

        print(
            f"{name:<30}"
            f"{_mean(forward_correct, key):>20.2f}"
            f"{_mean(backward_correct, key):>20.2f}"
        )