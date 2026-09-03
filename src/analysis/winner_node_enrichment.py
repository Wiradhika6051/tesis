from collections import Counter


def _to_counter(
    value,
):
    """
    Convert supported representations into Counter.
    """

    if value is None:

        return Counter()

    if isinstance(
        value,
        Counter,
    ):

        return Counter(
            value
        )

    if isinstance(
        value,
        dict,
    ):

        return Counter(
            value
        )

    if isinstance(
        value,
        (list, tuple, set),
    ):

        counter = Counter()

        for item in value:

            #
            # Already a node type string.
            #
            if isinstance(
                item,
                str,
            ):

                counter[
                    item
                ] += 1

            #
            # Dictionary node representation.
            #
            elif isinstance(
                item,
                dict,
            ):

                node_type = item.get(
                    "node_type"
                )

                if node_type is not None:

                    counter[
                        node_type
                    ] += 1

        return counter

    return Counter()


def _percentage(
    value,
    total,
):
    """
    Calculate percentage safely.
    """

    if total == 0:

        return 0.0

    return (
        value
        / total
        * 100
    )


def _get_winner_loser_types(
    record,
):
    """
    Select unique context node types belonging
    to the winning and losing directions.
    """

    outcome = record.get(
        "outcome"
    )

    forward_types = _to_counter(
        record.get(
            "forward_only_types"
        )
    )

    backward_types = _to_counter(
        record.get(
            "backward_only_types"
        )
    )

    if outcome == "FORWARD_CORRECT":

        return (
            forward_types,
            backward_types,
        )

    if outcome == "BACKWARD_CORRECT":

        return (
            backward_types,
            forward_types,
        )

    return (
        Counter(),
        Counter(),
    )


def analyze_winner_node_enrichment(
    directional_results,
):
    """
    Compare node-type composition between
    winning and losing directional contexts.
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

    results = {}

    for (
        outcome_name,
        records,
    ) in [
        (
            "FORWARD_CORRECT",
            forward_correct,
        ),
        (
            "BACKWARD_CORRECT",
            backward_correct,
        ),
    ]:

        winner_counter = Counter()

        loser_counter = Counter()

        for record in records:

            (
                winner_types,
                loser_types,
            ) = _get_winner_loser_types(
                record
            )

            winner_counter.update(
                winner_types
            )

            loser_counter.update(
                loser_types
            )

        winner_total = sum(
            winner_counter.values()
        )

        loser_total = sum(
            loser_counter.values()
        )

        all_types = sorted(
            set(
                winner_counter
            )
            |
            set(
                loser_counter
            ),
            key=lambda node_type: (
                -
                (
                    winner_counter[
                        node_type
                    ]
                    +
                    loser_counter[
                        node_type
                    ]
                ),
                node_type,
            ),
        )

        rows = []

        for node_type in all_types:

            winner_count = (
                winner_counter[
                    node_type
                ]
            )

            loser_count = (
                loser_counter[
                    node_type
                ]
            )

            winner_percentage = (
                _percentage(
                    winner_count,
                    winner_total,
                )
            )

            loser_percentage = (
                _percentage(
                    loser_count,
                    loser_total,
                )
            )

            difference = (
                winner_percentage
                -
                loser_percentage
            )

            rows.append(
                {
                    "node_type": node_type,

                    "winner_count":
                        winner_count,

                    "loser_count":
                        loser_count,

                    "winner_percentage":
                        winner_percentage,

                    "loser_percentage":
                        loser_percentage,

                    "difference":
                        difference,
                }
            )

        results[
            outcome_name
        ] = {
            "samples":
                len(records),

            "winner_total":
                winner_total,

            "loser_total":
                loser_total,

            "rows":
                rows,
        }

    return results


def print_winner_node_enrichment(
    results,
):
    """
    Print winner vs loser node-type enrichment.
    """

    print()

    print(
        "=" * 100
    )

    print(
        "WINNER VS LOSER NODE-TYPE ENRICHMENT"
    )

    print(
        "=" * 100
    )

    for outcome in [
        "FORWARD_CORRECT",
        "BACKWARD_CORRECT",
    ]:

        result = results[
            outcome
        ]

        print()

        print(
            outcome
        )

        print(
            "-" * 100
        )

        print()

        print(
            "Winner = correctly predicting direction"
        )

        print()

        print(
            f"{'Node type':<25}"
            f"{'Winner':>10}"
            f"{'Loser':>10}"
            f"{'Winner %':>12}"
            f"{'Loser %':>12}"
            f"{'Diff':>12}"
        )

        print(
            "-" * 85
        )

        for row in result[
            "rows"
        ]:

            print(
                f"{row['node_type']:<25}"
                f"{row['winner_count']:>10}"
                f"{row['loser_count']:>10}"
                f"{row['winner_percentage']:>11.2f}%"
                f"{row['loser_percentage']:>11.2f}%"
                f"{row['difference']:>11.2f}%"
            )