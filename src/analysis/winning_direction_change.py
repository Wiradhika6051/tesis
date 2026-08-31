from collections import defaultdict
import statistics
from src.analysis.paired_slice_similarity import normalize_text


# ============================================================
# Helpers
# ============================================================

def _sample_id(sample):
    """
    Return the identifier used to match the same sample
    across forward and backward representations.
    """

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label
    )


def _pair_id(sample):
    """
    Return the identifier shared by vulnerable and fixed
    versions of the same change.
    """

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path
    )


def _node_signature(node):
    """
    Semantic node representation.

    Node IDs and line numbers are intentionally excluded.
    """

    return (
        node.node_type,
        normalize_text(
            node.text
        )
    )


def _pruned_signatures(sample):
    """
    Return semantic signatures contained in the pruned CFG.
    """

    if sample.pruned_cfg is None:
        return set()

    return {

        _node_signature(node)

        for node in sample.pruned_cfg["nodes"]

    }


def _group_by_pair(samples):
    """
    Group samples by vulnerable/fixed pair.

    Returns:

        {
            pair_id: {
                1: vulnerable_sample,
                0: fixed_sample
            }
        }
    """

    grouped = defaultdict(dict)

    for sample in samples:

        grouped[
            _pair_id(sample)
        ][
            sample.label
        ] = sample

    return grouped


def _change_sets(
    vulnerable,
    fixed
):
    """
    Determine semantic nodes that are unique to either
    vulnerable or fixed version.

    Returns:

        vulnerable_only
        fixed_only
        unchanged
    """

    vulnerable_nodes = _pruned_signatures(
        vulnerable
    )

    fixed_nodes = _pruned_signatures(
        fixed
    )

    vulnerable_only = (
        vulnerable_nodes
        -
        fixed_nodes
    )

    fixed_only = (
        fixed_nodes
        -
        vulnerable_nodes
    )

    unchanged = (
        vulnerable_nodes
        &
        fixed_nodes
    )

    return (
        vulnerable_only,
        fixed_only,
        unchanged
    )


def _safe_ratio(
    numerator,
    denominator
):

    if denominator == 0:
        return 0.0

    return (
        numerator
        /
        denominator
    )


# ============================================================
# Main analysis
# ============================================================

def analyze_winning_direction_change(
    directional_results,
    forward_samples,
    backward_samples
):
    """
    Analyze whether the direction that correctly predicts
    a sample captures more of the semantic change than the
    losing direction.

    Only disagreement cases are analyzed.

    Outcomes:

        FORWARD_CORRECT
        BACKWARD_CORRECT
    """

    #
    # --------------------------------------------------------
    # Build sample lookup.
    # --------------------------------------------------------
    #

    forward_lookup = {

        _sample_id(sample):
            sample

        for sample in forward_samples

    }

    backward_lookup = {

        _sample_id(sample):
            sample

        for sample in backward_samples

    }

    #
    # --------------------------------------------------------
    # Build pair lookup.
    #
    # This lets us compare the vulnerable and fixed versions
    # of the same change.
    # --------------------------------------------------------
    #

    forward_pairs = _group_by_pair(
        forward_samples
    )

    backward_pairs = _group_by_pair(
        backward_samples
    )

    records = []

    unmatched = []

    #
    # --------------------------------------------------------
    # Process only prediction disagreements.
    # --------------------------------------------------------
    #

    for result in directional_results:

        outcome = result.get(
            "outcome"
        )

        if outcome not in (
            "FORWARD_CORRECT",
            "BACKWARD_CORRECT"
        ):

            continue

        sample_id = result.get(
            "sample_id"
        )

        if sample_id is None:
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

            unmatched.append({

                "sample_id":
                    sample_id,

                "reason":
                    "missing_direction"

            })

            continue

        #
        # Pair identifier.
        #

        pair_id = (
            sample_id[0],
            sample_id[1],
            sample_id[2]
        )

        #
        # Need vulnerable and fixed samples.
        #

        forward_pair = forward_pairs.get(
            pair_id,
            {}
        )

        backward_pair = backward_pairs.get(
            pair_id,
            {}
        )

        forward_vulnerable = forward_pair.get(
            1
        )

        forward_fixed = forward_pair.get(
            0
        )

        backward_vulnerable = backward_pair.get(
            1
        )

        backward_fixed = backward_pair.get(
            0
        )

        if (
            forward_vulnerable is None
            or
            forward_fixed is None
            or
            backward_vulnerable is None
            or
            backward_fixed is None
        ):

            unmatched.append({

                "sample_id":
                    sample_id,

                "reason":
                    "incomplete_vulnerable_fixed_pair"

            })

            continue

        #
        # ----------------------------------------------------
        # Semantic change for each direction.
        # ----------------------------------------------------
        #

        (
            forward_vulnerable_change,
            forward_fixed_change,
            forward_unchanged

        ) = _change_sets(

            forward_vulnerable,
            forward_fixed

        )

        (
            backward_vulnerable_change,
            backward_fixed_change,
            backward_unchanged

        ) = _change_sets(

            backward_vulnerable,
            backward_fixed

        )

        #
        # ----------------------------------------------------
        # Current sample's slice.
        #
        # Important:
        # The sample itself is either vulnerable or fixed.
        # ----------------------------------------------------
        #

        forward_slice = _pruned_signatures(
            forward_sample
        )

        backward_slice = _pruned_signatures(
            backward_sample
        )

        #
        # ----------------------------------------------------
        # Changed nodes captured by each direction.
        #
        # A changed node means it exists only in one version.
        # ----------------------------------------------------
        #

        forward_changed = (
            forward_vulnerable_change
            |
            forward_fixed_change
        )

        backward_changed = (
            backward_vulnerable_change
            |
            backward_fixed_change
        )

        forward_changed_captured = (
            forward_slice
            &
            forward_changed
        )

        backward_changed_captured = (
            backward_slice
            &
            backward_changed
        )

        #
        # ----------------------------------------------------
        # Directional-only nodes.
        # ----------------------------------------------------
        #

        forward_only = (
            forward_slice
            -
            backward_slice
        )

        backward_only = (
            backward_slice
            -
            forward_slice
        )

        #
        # Changed nodes inside directional-only context.
        # ----------------------------------------------------
        #

        forward_only_changed = (
            forward_only
            &
            forward_changed
        )

        backward_only_changed = (
            backward_only
            &
            backward_changed
        )

        #
        # ----------------------------------------------------
        # Coverage.
        # ----------------------------------------------------
        #

        forward_change_coverage = _safe_ratio(

            len(
                forward_changed_captured
            ),

            len(
                forward_changed
            )

        )

        backward_change_coverage = _safe_ratio(

            len(
                backward_changed_captured
            ),

            len(
                backward_changed
            )

        )

        #
        # ----------------------------------------------------
        # Does the winning direction capture more change?
        # ----------------------------------------------------
        #

        if (
            forward_change_coverage
            >
            backward_change_coverage
        ):

            change_winner = "FORWARD"

        elif (
            backward_change_coverage
            >
            forward_change_coverage
        ):

            change_winner = "BACKWARD"

        else:

            change_winner = "TIE"

        #
        # ----------------------------------------------------
        # Record.
        # ----------------------------------------------------
        #

        records.append({

            "sample_id":
                sample_id,

            "pair_id":
                pair_id,

            "outcome":
                outcome,

            #
            # Slice sizes.
            #

            "forward_size":
                len(forward_slice),

            "backward_size":
                len(backward_slice),

            "forward_only":
                len(forward_only),

            "backward_only":
                len(backward_only),

            #
            # Semantic change.
            #

            "forward_changed":
                len(forward_changed),

            "backward_changed":
                len(backward_changed),

            "forward_changed_captured":
                len(forward_changed_captured),

            "backward_changed_captured":
                len(backward_changed_captured),

            #
            # Directional-only changed content.
            #

            "forward_only_changed":
                len(forward_only_changed),

            "backward_only_changed":
                len(backward_only_changed),

            #
            # Coverage.
            #

            "forward_change_coverage":
                forward_change_coverage,

            "backward_change_coverage":
                backward_change_coverage,

            "change_coverage_difference":
                (
                    forward_change_coverage
                    -
                    backward_change_coverage
                ),

            "change_winner":
                change_winner

        })

    return {

        "records":
            records,

        "unmatched":
            unmatched

    }


# ============================================================
# Summary
# ============================================================

def summarize_winning_direction_change(
    analysis
):

    records = analysis[
        "records"
    ]

    outcomes = [

        "FORWARD_CORRECT",
        "BACKWARD_CORRECT"

    ]

    summary = {}

    for outcome in outcomes:

        subset = [

            r

            for r in records

            if r["outcome"] == outcome

        ]

        if not subset:

            summary[outcome] = {

                "samples":
                    0

            }

            continue

        forward_coverage = [

            r["forward_change_coverage"]

            for r in subset

        ]

        backward_coverage = [

            r["backward_change_coverage"]

            for r in subset

        ]

        summary[outcome] = {

            "samples":
                len(subset),

            "forward_change_coverage":
                statistics.mean(
                    forward_coverage
                ),

            "backward_change_coverage":
                statistics.mean(
                    backward_coverage
                ),

            "coverage_difference":
                statistics.mean([

                    r[
                        "change_coverage_difference"
                    ]

                    for r in subset

                ]),

            "forward_wins":
                sum(

                    r["change_winner"]
                    ==
                    "FORWARD"

                    for r in subset

                ),

            "backward_wins":
                sum(

                    r["change_winner"]
                    ==
                    "BACKWARD"

                    for r in subset

                ),

            "ties":
                sum(

                    r["change_winner"]
                    ==
                    "TIE"

                    for r in subset

                ),

            "average_forward_only_changed":
                statistics.mean([

                    r[
                        "forward_only_changed"
                    ]

                    for r in subset

                ]),

            "average_backward_only_changed":
                statistics.mean([

                    r[
                        "backward_only_changed"
                    ]

                    for r in subset

                ])

        }

    return summary


# ============================================================
# Printer
# ============================================================

def print_winning_direction_change(
    analysis
):

    records = analysis[
        "records"
    ]

    summary = summarize_winning_direction_change(
        analysis
    )

    print()

    print(
        "=" * 100
    )

    print(
        "WINNING DIRECTION VS SEMANTIC CHANGE"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"{'Metric':<40}"
        f"{'Forward correct':>20}"
        f"{'Backward correct':>20}"
    )

    print(
        "-" * 80
    )

    for metric, key in [

        (
            "Samples",
            "samples"
        ),

        (
            "Forward change coverage",
            "forward_change_coverage"
        ),

        (
            "Backward change coverage",
            "backward_change_coverage"
        ),

        (
            "Coverage difference",
            "coverage_difference"
        ),

        (
            "Forward change wins",
            "forward_wins"
        ),

        (
            "Backward change wins",
            "backward_wins"
        ),

        (
            "Ties",
            "ties"
        ),

        (
            "Forward-only changed nodes",
            "average_forward_only_changed"
        ),

        (
            "Backward-only changed nodes",
            "average_backward_only_changed"
        )

    ]:

        forward = summary[
            "FORWARD_CORRECT"
        ].get(
            key,
            0
        )

        backward = summary[
            "BACKWARD_CORRECT"
        ].get(
            key,
            0
        )

        if "coverage" in metric.lower() \
           or "difference" in metric.lower():

            print(

                f"{metric:<40}"
                f"{forward * 100:>19.2f}%"
                f"{backward * 100:>19.2f}%"

            )

        else:

            print(

                f"{metric:<40}"
                f"{forward:>20.2f}"
                f"{backward:>20.2f}"

            )

    #
    # --------------------------------------------------------
    # Per-sample results.
    # --------------------------------------------------------
    #

    print()

    print(
        "=" * 100
    )

    print(
        "PER-SAMPLE CHANGE CAPTURE"
    )

    print(
        "=" * 100
    )

    for r in records:

        print()

        print(
            "Outcome:",
            r["outcome"]
        )

        print(
            "Sample:",
            r["sample_id"]
        )

        print(
            "Forward coverage:",
            f"{r['forward_change_coverage'] * 100:.2f}%"
        )

        print(
            "Backward coverage:",
            f"{r['backward_change_coverage'] * 100:.2f}%"
        )

        print(
            "Forward-only changed:",
            r["forward_only_changed"]
        )

        print(
            "Backward-only changed:",
            r["backward_only_changed"]
        )

        print(
            "Change winner:",
            r["change_winner"]
        )

    #
    # --------------------------------------------------------
    # Unmatched.
    # --------------------------------------------------------
    #

    print()

    print(
        "Unmatched:",
        len(
            analysis["unmatched"]
        )
    )