from collections import Counter


def analyze_directional_context_presence(
    winner_context_results,
):
    """
    Analyze whether winner and loser directions
    contain unique directional context.

    Context states:

    BOTH_NONEMPTY
        Winner and loser both contain
        directional-only context.

    WINNER_ONLY
        Only the winning direction contains
        directional-only context.

    LOSER_ONLY
        Only the losing direction contains
        directional-only context.

    BOTH_EMPTY
        Neither direction contains
        directional-only context.
    """

    results = {}

    for outcome, records in winner_context_results.items():

        counts = Counter()

        per_sample = []

        for record in records:

            winner_context = record.get(
                "winner_context",
                []
            )

            loser_context = record.get(
                "loser_context",
                []
            )

            winner_nonempty = bool(
                winner_context
            )

            loser_nonempty = bool(
                loser_context
            )

            if (
                winner_nonempty
                and loser_nonempty
            ):
                state = "BOTH_NONEMPTY"

            elif winner_nonempty:
                state = "WINNER_ONLY"

            elif loser_nonempty:
                state = "LOSER_ONLY"

            else:
                state = "BOTH_EMPTY"

            counts[state] += 1

            per_sample.append(
                {
                    "sample_id": record.get(
                        "sample_id"
                    ),
                    "winner": record.get(
                        "winner"
                    ),
                    "state": state,
                    "winner_context_size": len(
                        winner_context
                    ),
                    "loser_context_size": len(
                        loser_context
                    ),
                }
            )

        total = len(records)

        percentages = {}

        for state in [
            "BOTH_NONEMPTY",
            "WINNER_ONLY",
            "LOSER_ONLY",
            "BOTH_EMPTY",
        ]:

            count = counts[state]

            if total > 0:

                percentage = (
                    count / total
                ) * 100

            else:

                percentage = 0.0

            percentages[state] = percentage

        results[outcome] = {
            "total_samples": total,
            "counts": dict(
                counts
            ),
            "percentages": percentages,
            "per_sample": per_sample,
        }

    return results


def print_directional_context_presence(
    results,
):
    """
    Print directional context presence analysis.
    """

    print()
    print("=" * 80)
    print("DIRECTIONAL CONTEXT PRESENCE")
    print("=" * 80)

    states = [
        (
            "BOTH_NONEMPTY",
            "Both directions contain unique context",
        ),
        (
            "WINNER_ONLY",
            "Only winning direction contains unique context",
        ),
        (
            "LOSER_ONLY",
            "Only losing direction contains unique context",
        ),
        (
            "BOTH_EMPTY",
            "Neither direction contains unique context",
        ),
    ]

    for outcome, data in results.items():

        print()
        print("-" * 80)
        print(outcome)
        print("-" * 80)

        total = data[
            "total_samples"
        ]

        print(
            f"Total samples: {total}"
        )

        print()

        print(
            f"{'Context state':<55}"
            f"{'Count':>8}"
            f"{'Percent':>12}"
        )

        print("-" * 80)

        for state, description in states:

            count = data[
                "counts"
            ].get(
                state,
                0
            )

            percentage = data[
                "percentages"
            ].get(
                state,
                0.0
            )

            print(
                f"{description:<55}"
                f"{count:>8}"
                f"{percentage:>11.2f}%"
            )