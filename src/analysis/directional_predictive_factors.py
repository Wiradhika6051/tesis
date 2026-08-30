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

    forward_smaller = sum(

        r["forward_slice"]
        <
        r["backward_slice"]

        for r in forward_wins

    )

    backward_smaller = sum(

        r["backward_slice"]
        <
        r["forward_slice"]

        for r in backward_wins

    )

    equal_forward = sum(

        r["forward_slice"]
        ==
        r["backward_slice"]

        for r in forward_wins

    )

    equal_backward = sum(

        r["forward_slice"]
        ==
        r["backward_slice"]

        for r in backward_wins

    )

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
        forward_smaller
    )

    print(
        "  Equal:",
        equal_forward
    )

    print()

    print(
        "Backward-correct samples:",
        len(backward_wins)
    )

    print(
        "  Backward slice smaller:",
        backward_smaller
    )

    print(
        "  Equal:",
        equal_backward
    )