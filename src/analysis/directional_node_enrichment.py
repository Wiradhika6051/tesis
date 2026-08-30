from collections import Counter, defaultdict
import statistics


OUTCOMES = [
    "FORWARD_CORRECT",
    "BACKWARD_CORRECT"
]


def _node_type_from_signature(
    signature
):

    #
    # get_node_signature() returns:
    #
    # (
    #     node.node_type,
    #     normalize_text(node.text)
    # )
    #
    return signature[0]


def _safe_ratio(
    numerator,
    denominator
):

    if denominator == 0:
        return 0.0

    return numerator / denominator


def analyze_directional_node_enrichment(
    composition_results
):
    """
    Analyze node-type enrichment in the directional
    disagreement cases.

    Only FORWARD_CORRECT and BACKWARD_CORRECT samples
    are analyzed.

    The analysis compares node types appearing in:

        forward-only

        backward-only

    and normalizes their frequencies by the total
    number of directional-only nodes.
    """

    grouped = defaultdict(list)

    for record in composition_results:

        outcome = record.get(
            "outcome"
        )

        if outcome in OUTCOMES:

            grouped[
                outcome
            ].append(
                record
            )

    results = {}

    for outcome in OUTCOMES:

        records = grouped[
            outcome
        ]

        #
        # Aggregate node types.
        #
        forward_types = Counter()
        backward_types = Counter()

        for record in records:

            forward_types.update(
                record.get(
                    "forward_only_types",
                    {}
                )
            )

            backward_types.update(
                record.get(
                    "backward_only_types",
                    {}
                )
            )

        forward_total = sum(
            forward_types.values()
        )

        backward_total = sum(
            backward_types.values()
        )

        #
        # Convert to normalized frequency.
        #
        forward_frequency = {

            node_type:
                _safe_ratio(
                    count,
                    forward_total
                )

            for node_type, count
            in forward_types.items()

        }

        backward_frequency = {

            node_type:
                _safe_ratio(
                    count,
                    backward_total
                )

            for node_type, count
            in backward_types.items()

        }

        results[
            outcome
        ] = {

            "samples":
                len(records),

            "forward_types":
                forward_types,

            "backward_types":
                backward_types,

            "forward_total":
                forward_total,

            "backward_total":
                backward_total,

            "forward_frequency":
                forward_frequency,

            "backward_frequency":
                backward_frequency

        }

    return results


def print_directional_node_enrichment(
    results,
    top_n=20
):

    print()

    print(
        "=" * 80
    )

    print(
        "DIRECTIONAL NODE-TYPE ENRICHMENT"
    )

    print(
        "=" * 80
    )

    for outcome in OUTCOMES:

        data = results.get(
            outcome
        )

        if data is None:
            continue

        print()

        print(
            outcome
        )

        print(
            "-" * 60
        )

        print(
            "Samples:",
            data["samples"]
        )

        print()

        print(
            "FORWARD-ONLY"
        )

        print(
            f"{'Node type':<25}"
            f"{'Count':>10}"
            f"{'Frequency':>15}"
        )

        print(
            "-" * 50
        )

        items = sorted(

            data[
                "forward_frequency"
            ].items(),

            key=lambda x: x[1],

            reverse=True

        )

        for node_type, frequency in items[:top_n]:

            count = data[
                "forward_types"
            ][
                node_type
            ]

            print(

                f"{node_type:<25}"
                f"{count:>10}"
                f"{frequency * 100:>14.2f}%"

            )

        print()

        print(
            "BACKWARD-ONLY"
        )

        print(
            f"{'Node type':<25}"
            f"{'Count':>10}"
            f"{'Frequency':>15}"
        )

        print(
            "-" * 50
        )

        items = sorted(

            data[
                "backward_frequency"
            ].items(),

            key=lambda x: x[1],

            reverse=True

        )

        for node_type, frequency in items[:top_n]:

            count = data[
                "backward_types"
            ][
                node_type
            ]

            print(

                f"{node_type:<25}"
                f"{count:>10}"
                f"{frequency * 100:>14.2f}%"

            )


def compare_correct_directions(
    results,
    top_n=30
):

    forward = results.get(
        "FORWARD_CORRECT",
        {}
    )

    backward = results.get(
        "BACKWARD_CORRECT",
        {}
    )

    forward_freq = forward.get(
        "forward_frequency",
        {}
    )

    backward_freq = backward.get(
        "backward_frequency",
        {}
    )

    #
    # Compare the node-type distributions
    # for the directional side that was correct.
    #
    all_types = set(
        forward_freq
    ) | set(
        backward_freq
    )

    rows = []

    for node_type in all_types:

        forward_value = (
            forward_freq.get(
                node_type,
                0.0
            )
        )

        backward_value = (
            backward_freq.get(
                node_type,
                0.0
            )
        )

        rows.append({

            "node_type":
                node_type,

            "forward_correct":
                forward_value,

            "backward_correct":
                backward_value,

            "difference":
                forward_value
                -
                backward_value

        })

    rows.sort(

        key=lambda row:
            abs(
                row["difference"]
            ),

        reverse=True

    )

    print()

    print(
        "=" * 80
    )

    print(
        "FORWARD-CORRECT VS BACKWARD-CORRECT NODE TYPES"
    )

    print(
        "=" * 80
    )

    print()

    print(

        f"{'Node type':<25}"
        f"{'Forward correct':>18}"
        f"{'Backward correct':>18}"
        f"{'Difference':>15}"

    )

    print(
        "-" * 80
    )

    for row in rows[:top_n]:

        print(

            f"{row['node_type']:<25}"
            f"{row['forward_correct'] * 100:>17.2f}%"
            f"{row['backward_correct'] * 100:>17.2f}%"
            f"{row['difference'] * 100:>14.2f}%"

        )

    return rows