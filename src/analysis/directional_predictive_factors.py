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