from collections import defaultdict
import statistics


OUTCOMES = [
    "BOTH_CORRECT",
    "BOTH_WRONG",
    "FORWARD_CORRECT",
    "BACKWARD_CORRECT"
]


def _mean(records, key):

    values = [
        record[key]
        for record in records
        if record[key] is not None
    ]

    if not values:
        return 0.0

    return statistics.mean(values)


def _median(records, key):

    values = [
        record[key]
        for record in records
        if record[key] is not None
    ]

    if not values:
        return 0.0

    return statistics.median(values)


def _safe_ratio(
    numerator,
    denominator
):

    if denominator == 0:
        return 0.0

    return numerator / denominator


def build_context_profiles(
    directional_results
):
    """
    Build a sample-level directional context profile.

    Expected fields in directional_results:

        forward_slice
        backward_slice
        forward_only
        backward_only
        overlap

    The analysis derives:

        directional ratios
        local context ratios
        distant context ratios
        forward/backward context balance
    """

    profiles = []

    for result in directional_results:

        outcome = result.get(
            "outcome"
        )

        if outcome not in OUTCOMES:
            continue

        #
        # Basic slice sizes.
        #
        forward_size = result.get(
            "forward_slice",
            result.get(
                "forward_size",
                0
            )
        )

        backward_size = result.get(
            "backward_slice",
            result.get(
                "backward_size",
                0
            )
        )

        forward_only = result.get(
            "forward_only",
            0
        )

        backward_only = result.get(
            "backward_only",
            0
        )

        overlap = result.get(
            "overlap",
            0
        )

        #
        # Some implementations store the
        # actual sets instead of counts.
        #
        if isinstance(
            forward_only,
            (set, list, tuple)
        ):

            forward_only = len(
                forward_only
            )

        if isinstance(
            backward_only,
            (set, list, tuple)
        ):

            backward_only = len(
                backward_only
            )

        if isinstance(
            overlap,
            (set, list, tuple)
        ):

            overlap = len(
                overlap
            )

        #
        # Total directional context.
        #
        directional_total = (
            forward_only
            +
            backward_only
        )

        #
        # Ratios relative to function/slice context.
        #
        forward_only_ratio = _safe_ratio(
            forward_only,
            forward_size
        )

        backward_only_ratio = _safe_ratio(
            backward_only,
            backward_size
        )

        #
        # Relative directional dominance.
        #
        forward_context_ratio = _safe_ratio(
            forward_only,
            directional_total
        )

        backward_context_ratio = _safe_ratio(
            backward_only,
            directional_total
        )

        #
        # Balance.
        #
        context_difference = (
            forward_only
            -
            backward_only
        )

        context_ratio = _safe_ratio(
            forward_only,
            backward_only
        )

        profiles.append({

            "sample_id":
                result.get(
                    "sample_id"
                ),

            "outcome":
                outcome,

            "forward_slice":
                forward_size,

            "backward_slice":
                backward_size,

            "forward_only":
                forward_only,

            "backward_only":
                backward_only,

            "overlap":
                overlap,

            "directional_total":
                directional_total,

            "forward_only_ratio":
                forward_only_ratio,

            "backward_only_ratio":
                backward_only_ratio,

            "forward_context_ratio":
                forward_context_ratio,

            "backward_context_ratio":
                backward_context_ratio,

            "context_difference":
                context_difference,

            "context_ratio":
                context_ratio,

            #
            # These can be populated by the
            # distance analysis if available.
            #
            "forward_local":
                result.get(
                    "forward_local",
                    0
                ),

            "backward_local":
                result.get(
                    "backward_local",
                    0
                ),

            "forward_distant":
                result.get(
                    "forward_distant",
                    0
                ),

            "backward_distant":
                result.get(
                    "backward_distant",
                    0
                )

        })

    return profiles


def summarize_context_profiles(
    profiles
):

    grouped = defaultdict(list)

    for profile in profiles:

        grouped[
            profile["outcome"]
        ].append(
            profile
        )

    summaries = {}

    for outcome in OUTCOMES:

        records = grouped.get(
            outcome,
            []
        )

        if not records:

            summaries[
                outcome
            ] = {

                "samples": 0

            }

            continue

        summaries[
            outcome
        ] = {

            "samples":
                len(records),

            "forward_slice":
                _mean(
                    records,
                    "forward_slice"
                ),

            "backward_slice":
                _mean(
                    records,
                    "backward_slice"
                ),

            "forward_only":
                _mean(
                    records,
                    "forward_only"
                ),

            "backward_only":
                _mean(
                    records,
                    "backward_only"
                ),

            "overlap":
                _mean(
                    records,
                    "overlap"
                ),

            "forward_only_ratio":
                _mean(
                    records,
                    "forward_only_ratio"
                ),

            "backward_only_ratio":
                _mean(
                    records,
                    "backward_only_ratio"
                ),

            "forward_context_ratio":
                _mean(
                    records,
                    "forward_context_ratio"
                ),

            "backward_context_ratio":
                _mean(
                    records,
                    "backward_context_ratio"
                ),

            "context_difference":
                _mean(
                    records,
                    "context_difference"
                ),

            "context_ratio":
                _mean(
                    records,
                    "context_ratio"
                ),

            "forward_local":
                _mean(
                    records,
                    "forward_local"
                ),

            "backward_local":
                _mean(
                    records,
                    "backward_local"
                ),

            "forward_distant":
                _mean(
                    records,
                    "forward_distant"
                ),

            "backward_distant":
                _mean(
                    records,
                    "backward_distant"
                )

        }

    return summaries


def print_context_profiles(
    summaries
):

    print()

    print(
        "=" * 100
    )

    print(
        "DIRECTIONAL CONTEXT PROFILE BY OUTCOME"
    )

    print(
        "=" * 100
    )

    headers = [

        "Metric",
        "Both correct",
        "Both wrong",
        "Forward correct",
        "Backward correct"

    ]

    print()

    print(

        f"{headers[0]:<32}"
        f"{headers[1]:>16}"
        f"{headers[2]:>16}"
        f"{headers[3]:>18}"
        f"{headers[4]:>19}"

    )

    print(
        "-" * 105
    )

    def row(
        name,
        key,
        percent=False
    ):

        values = []

        for outcome in OUTCOMES:

            value = summaries.get(
                outcome,
                {}
            ).get(
                key,
                0.0
            )

            if percent:

                values.append(
                    f"{value * 100:.2f}%"
                )

            else:

                values.append(
                    f"{value:.2f}"
                )

        print(

            f"{name:<32}"
            f"{values[0]:>16}"
            f"{values[1]:>16}"
            f"{values[2]:>18}"
            f"{values[3]:>19}"

        )

    row(
        "Forward slice",
        "forward_slice"
    )

    row(
        "Backward slice",
        "backward_slice"
    )

    print()

    row(
        "Forward-only",
        "forward_only"
    )

    row(
        "Backward-only",
        "backward_only"
    )

    row(
        "Overlap",
        "overlap"
    )

    print()

    row(
        "Forward-only ratio",
        "forward_only_ratio",
        percent=True
    )

    row(
        "Backward-only ratio",
        "backward_only_ratio",
        percent=True
    )

    row(
        "Forward context share",
        "forward_context_ratio",
        percent=True
    )

    row(
        "Backward context share",
        "backward_context_ratio",
        percent=True
    )

    print()

    row(
        "Forward - backward context",
        "context_difference"
    )

    row(
        "Forward / backward context",
        "context_ratio"
    )

    print()

    row(
        "Forward local",
        "forward_local",
        percent=True
    )

    row(
        "Backward local",
        "backward_local",
        percent=True
    )

    row(
        "Forward distant",
        "forward_distant",
        percent=True
    )

    row(
        "Backward distant",
        "backward_distant",
        percent=True
    )

def print_extreme_context_cases(
    profiles,
    limit=20
):

    #
    # Sort by absolute directional imbalance.
    #
    records = sorted(

        profiles,

        key=lambda record:
            abs(
                record[
                    "context_difference"
                ]
            ),

        reverse=True

    )

    print()

    print(
        "=" * 100
    )

    print(
        "EXTREME DIRECTIONAL CONTEXT CASES"
    )

    print(
        "=" * 100
    )

    for record in records[:limit]:

        print()

        print(
            "Outcome:",
            record["outcome"]
        )

        print(
            "Sample ID:",
            record["sample_id"]
        )

        print(
            "Forward slice:",
            record["forward_slice"]
        )

        print(
            "Backward slice:",
            record["backward_slice"]
        )

        print(
            "Forward-only:",
            record["forward_only"]
        )

        print(
            "Backward-only:",
            record["backward_only"]
        )

        print(
            "Overlap:",
            record["overlap"]
        )

        print(
            "Context difference:",
            f"{record['context_difference']:.2f}"
        )

        print(
            "Forward context share:",
            f"{record['forward_context_ratio'] * 100:.2f}%"
        )

        print(
            "Backward context share:",
            f"{record['backward_context_ratio'] * 100:.2f}%"
        )