import statistics


def analyze_winning_direction(
    directional_results
):

    records = [

        result

        for result in directional_results

        if result.get("outcome")
        in (
            "FORWARD_CORRECT",
            "BACKWARD_CORRECT"
        )

    ]

    groups = {

        "FORWARD_CORRECT": [],

        "BACKWARD_CORRECT": []

    }

    for record in records:

        groups[
            record["outcome"]
        ].append(
            record
        )

    metrics = [

        "forward_slice",

        "backward_slice",

        "forward_only",

        "backward_only",

        "overlap",

        "forward_share",

        "backward_share",

        "directional_jaccard",

        "size_difference",

        "size_ratio"

    ]

    result = {}

    for outcome in groups:

        group = groups[
            outcome
        ]

        result[
            outcome
        ] = {

            "samples":
                len(group)

        }

        for metric in metrics:

            values = [

                r[metric]

                for r in group

                if metric in r
                and r[metric] is not None

            ]

            if values:

                result[
                    outcome
                ][
                    metric + "_mean"
                ] = statistics.mean(
                    values
                )

                result[
                    outcome
                ][
                    metric + "_median"
                ] = statistics.median(
                    values
                )

            else:

                result[
                    outcome
                ][
                    metric + "_mean"
                ] = 0.0

                result[
                    outcome
                ][
                    metric + "_median"
                ] = 0.0

    return result
def analyze_winner_slice_compactness(
    directional_results
):

    disagreements = [

        r

        for r in directional_results

        if r.get("outcome")
        in (
            "FORWARD_CORRECT",
            "BACKWARD_CORRECT"
        )

    ]

    forward_wins = [

        r

        for r in disagreements

        if r["outcome"]
        == "FORWARD_CORRECT"

    ]

    backward_wins = [

        r

        for r in disagreements

        if r["outcome"]
        == "BACKWARD_CORRECT"

    ]

    #
    # --------------------------------------------------
    # Forward-correct samples.
    # --------------------------------------------------
    #

    forward_smaller = sum(

        r["forward_size"]
        <
        r["backward_size"]

        for r in forward_wins

    )

    forward_larger = sum(

        r["forward_size"]
        >
        r["backward_size"]

        for r in forward_wins

    )

    forward_equal = sum(

        r["forward_size"]
        ==
        r["backward_size"]

        for r in forward_wins

    )

    #
    # --------------------------------------------------
    # Backward-correct samples.
    # --------------------------------------------------
    #

    backward_smaller = sum(

        r["backward_size"]
        <
        r["forward_size"]

        for r in backward_wins

    )

    backward_larger = sum(

        r["backward_size"]
        >
        r["forward_size"]

        for r in backward_wins

    )

    backward_equal = sum(

        r["backward_size"]
        ==
        r["forward_size"]

        for r in backward_wins

    )

    #
    # --------------------------------------------------
    # Print.
    # --------------------------------------------------
    #

    print()

    print(
        "=" * 80
    )

    print(
        "WINNING DIRECTION VS SLICE COMPACTNESS"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Forward-correct samples:",
        len(forward_wins)
    )

    print(
        "  Forward slice smaller:",
        forward_smaller,
        f"({forward_smaller / len(forward_wins) * 100:.2f}%)"
        if forward_wins
        else "(0.00%)"
    )

    print(
        "  Forward slice larger:",
        forward_larger,
        f"({forward_larger / len(forward_wins) * 100:.2f}%)"
        if forward_wins
        else "(0.00%)"
    )

    print(
        "  Equal:",
        forward_equal,
        f"({forward_equal / len(forward_wins) * 100:.2f}%)"
        if forward_wins
        else "(0.00%)"
    )

    print()

    print(
        "Backward-correct samples:",
        len(backward_wins)
    )

    print(
        "  Backward slice smaller:",
        backward_smaller,
        f"({backward_smaller / len(backward_wins) * 100:.2f}%)"
        if backward_wins
        else "(0.00%)"
    )

    print(
        "  Backward slice larger:",
        backward_larger,
        f"({backward_larger / len(backward_wins) * 100:.2f}%)"
        if backward_wins
        else "(0.00%)"
    )

    print(
        "  Equal:",
        backward_equal,
        f"({backward_equal / len(backward_wins) * 100:.2f}%)"
        if backward_wins
        else "(0.00%)"
    )

    return {

        "forward_correct": {

            "samples":
                len(forward_wins),

            "smaller":
                forward_smaller,

            "larger":
                forward_larger,

            "equal":
                forward_equal

        },

        "backward_correct": {

            "samples":
                len(backward_wins),

            "smaller":
                backward_smaller,

            "larger":
                backward_larger,

            "equal":
                backward_equal

        }

    }

def print_winning_direction_analysis(
    analysis
):

    print()

    print(
        "=" * 100
    )

    print(
        "WINNING DIRECTION: CONTEXT FACTORS"
    )

    print(
        "=" * 100
    )

    forward = analysis[
        "FORWARD_CORRECT"
    ]

    backward = analysis[
        "BACKWARD_CORRECT"
    ]

    print()

    print(

        f"{'Metric':<30}"
        f"{'Forward correct':>20}"
        f"{'Backward correct':>20}"
        f"{'Difference':>20}"

    )

    print(
        "-" * 95
    )

    metrics = [

        (
            "Forward slice",
            "forward_slice_mean"
        ),

        (
            "Backward slice",
            "backward_slice_mean"
        ),

        (
            "Forward-only",
            "forward_only_mean"
        ),

        (
            "Backward-only",
            "backward_only_mean"
        ),

        (
            "Overlap",
            "overlap_mean"
        ),

        (
            "Forward context share",
            "forward_share_mean"
        ),

        (
            "Backward context share",
            "backward_share_mean"
        ),

        (
            "Directional Jaccard",
            "directional_jaccard_mean"
        ),

        (
            "Size difference",
            "size_difference_mean"
        ),

        (
            "Size ratio",
            "size_ratio_mean"
        )

    ]

    for name, key in metrics:

        f = forward.get(
            key,
            0.0
        )

        b = backward.get(
            key,
            0.0
        )

        difference = f - b

        if (
            "share" in key
            or
            "jaccard" in key
        ):

            print(

                f"{name:<30}"
                f"{f * 100:>19.2f}%"
                f"{b * 100:>19.2f}%"
                f"{difference * 100:>19.2f}%"

            )

        else:

            print(

                f"{name:<30}"
                f"{f:>20.2f}"
                f"{b:>20.2f}"
                f"{difference:>20.2f}"

            )

import statistics


def _safe_mean(values):
    """
    Return mean or 0 when no values exist.
    """

    if not values:
        return 0.0

    return statistics.mean(values)


def _get_value(record, key, default=0):
    """
    Safely retrieve a numeric field.
    """

    value = record.get(key, default)

    if value is None:
        return default

    return value


def analyze_directional_predictive_factors(
    directional_results,
):
    """
    Analyze structural factors associated with
    FORWARD_CORRECT and BACKWARD_CORRECT outcomes.
    """

    forward_correct = [
        record
        for record in directional_results
        if record.get("outcome")
        == "FORWARD_CORRECT"
    ]

    backward_correct = [
        record
        for record in directional_results
        if record.get("outcome")
        == "BACKWARD_CORRECT"
    ]

    metrics = {
        "forward_size": {
            "label": "Forward slice",
        },
        "backward_size": {
            "label": "Backward slice",
        },
        "forward_only": {
            "label": "Forward-only",
        },
        "backward_only": {
            "label": "Backward-only",
        },
        "overlap": {
            "label": "Overlap",
        },
        "size_difference": {
            "label": "Size difference",
        },
    }

    results = {
        "forward_correct": {
            "samples": len(forward_correct),
            "metrics": {},
        },
        "backward_correct": {
            "samples": len(backward_correct),
            "metrics": {},
        },
    }

    for metric_name in metrics:

        results[
            "forward_correct"
        ]["metrics"][metric_name] = _safe_mean(
            [
                _get_value(
                    record,
                    metric_name,
                )
                for record in forward_correct
            ]
        )

        results[
            "backward_correct"
        ]["metrics"][metric_name] = _safe_mean(
            [
                _get_value(
                    record,
                    metric_name,
                )
                for record in backward_correct
            ]
        )

    #
    # Context-share metrics.
    #
    for group_name, records in [
        (
            "forward_correct",
            forward_correct,
        ),
        (
            "backward_correct",
            backward_correct,
        ),
    ]:

        forward_share_values = []
        backward_share_values = []
        jaccard_values = []
        size_ratio_values = []

        for record in records:

            forward_size = _get_value(
                record,
                "forward_size",
            )

            backward_size = _get_value(
                record,
                "backward_size",
            )

            overlap = _get_value(
                record,
                "overlap",
            )

            forward_only = _get_value(
                record,
                "forward_only",
            )

            backward_only = _get_value(
                record,
                "backward_only",
            )

            #
            # Share of directional context.
            #
            if forward_size > 0:

                forward_share_values.append(
                    forward_only
                    / forward_size
                )

            if backward_size > 0:

                backward_share_values.append(
                    backward_only
                    / backward_size
                )

            #
            # Jaccard similarity.
            #
            union = (
                forward_only
                +
                backward_only
                +
                overlap
            )

            if union > 0:

                jaccard_values.append(
                    overlap
                    / union
                )

            #
            # Size ratio.
            #
            if backward_size > 0:

                size_ratio_values.append(
                    forward_size
                    / backward_size
                )

        results[
            group_name
        ]["metrics"][
            "forward_context_share"
        ] = _safe_mean(
            forward_share_values
        )

        results[
            group_name
        ]["metrics"][
            "backward_context_share"
        ] = _safe_mean(
            backward_share_values
        )

        results[
            group_name
        ]["metrics"][
            "directional_jaccard"
        ] = _safe_mean(
            jaccard_values
        )

        results[
            group_name
        ]["metrics"][
            "size_ratio"
        ] = _safe_mean(
            size_ratio_values
        )

    return results


def print_directional_predictive_factors(
    results,
):
    """
    Print directional predictive factor analysis.
    """

    print()

    print(
        "=" * 100
    )

    print(
        "WINNING DIRECTION: CONTEXT FACTORS"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"{'Metric':<35}"
        f"{'Forward correct':>20}"
        f"{'Backward correct':>20}"
        f"{'Difference':>20}"
    )

    print(
        "-" * 95
    )

    metric_labels = [
        (
            "forward_size",
            "Forward slice",
            False,
        ),
        (
            "backward_size",
            "Backward slice",
            False,
        ),
        (
            "forward_only",
            "Forward-only",
            False,
        ),
        (
            "backward_only",
            "Backward-only",
            False,
        ),
        (
            "overlap",
            "Overlap",
            False,
        ),
        (
            "forward_context_share",
            "Forward context share",
            True,
        ),
        (
            "backward_context_share",
            "Backward context share",
            True,
        ),
        (
            "directional_jaccard",
            "Directional Jaccard",
            True,
        ),
        (
            "size_difference",
            "Size difference",
            False,
        ),
        (
            "size_ratio",
            "Size ratio",
            False,
        ),
    ]

    forward_metrics = (
        results[
            "forward_correct"
        ]["metrics"]
    )

    backward_metrics = (
        results[
            "backward_correct"
        ]["metrics"]
    )

    for (
        metric,
        label,
        is_percentage,
    ) in metric_labels:

        forward_value = (
            forward_metrics.get(
                metric,
                0.0,
            )
        )

        backward_value = (
            backward_metrics.get(
                metric,
                0.0,
            )
        )

        difference = (
            forward_value
            -
            backward_value
        )

        if is_percentage:

            forward_text = (
                f"{forward_value:.2%}"
            )

            backward_text = (
                f"{backward_value:.2%}"
            )

            difference_text = (
                f"{difference:.2%}"
            )

        else:

            forward_text = (
                f"{forward_value:.2f}"
            )

            backward_text = (
                f"{backward_value:.2f}"
            )

            difference_text = (
                f"{difference:.2f}"
            )

        print(
            f"{label:<35}"
            f"{forward_text:>20}"
            f"{backward_text:>20}"
            f"{difference_text:>20}"
        )