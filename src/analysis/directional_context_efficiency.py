from collections import defaultdict
import statistics


OUTCOMES = [
    "BOTH_CORRECT",
    "BOTH_WRONG",
    "FORWARD_CORRECT",
    "BACKWARD_CORRECT"
]


def _safe_ratio(
    numerator,
    denominator
):

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _mean(
    records,
    key
):

    values = [

        record[key]

        for record in records

        if record[key] is not None

    ]

    if not values:
        return 0.0

    return statistics.mean(values)


def _median(
    records,
    key
):

    values = [

        record[key]

        for record in records

        if record[key] is not None

    ]

    if not values:
        return 0.0

    return statistics.median(values)


def build_context_efficiency_profiles(
    directional_results
):

    profiles = []

    for result in directional_results:

        outcome = result.get(
            "outcome"
        )

        if outcome not in OUTCOMES:
            continue

        #
        # Directional components.
        #
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
        # Derive complete slice sizes.
        #
        forward = (
            forward_only
            +
            overlap
        )

        backward = (
            backward_only
            +
            overlap
        )

        #
        # Union.
        #
        union = (
            forward_only
            +
            backward_only
            +
            overlap
        )

        directional_total = (
            forward_only
            +
            backward_only
        )

        #
        # Directional dominance.
        #
        forward_share = _safe_ratio(
            forward_only,
            directional_total
        )

        backward_share = _safe_ratio(
            backward_only,
            directional_total
        )

        #
        # Similarity between directional slices.
        #
        directional_jaccard = _safe_ratio(
            overlap,
            union
        )

        #
        # Size difference.
        #
        size_difference = (
            forward
            -
            backward
        )

        size_ratio = _safe_ratio(
            forward,
            backward
        )

        profiles.append({

            "sample_id":
                result.get(
                    "sample_id"
                ),

            "outcome":
                outcome,

            "forward_slice":
                forward,

            "backward_slice":
                backward,

            "forward_only":
                forward_only,

            "backward_only":
                backward_only,

            "overlap":
                overlap,

            "union":
                union,

            "forward_share":
                forward_share,

            "backward_share":
                backward_share,

            "directional_jaccard":
                directional_jaccard,

            "size_difference":
                size_difference,

            "size_ratio":
                size_ratio

        })

    return profiles


def summarize_context_efficiency(
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

        summaries[
            outcome
        ] = {

            "samples":
                len(records),

            "forward_slice_mean":
                _mean(
                    records,
                    "forward_slice"
                ),

            "forward_slice_median":
                _median(
                    records,
                    "forward_slice"
                ),

            "backward_slice_mean":
                _mean(
                    records,
                    "backward_slice"
                ),

            "backward_slice_median":
                _median(
                    records,
                    "backward_slice"
                ),

            "forward_only_mean":
                _mean(
                    records,
                    "forward_only"
                ),

            "backward_only_mean":
                _mean(
                    records,
                    "backward_only"
                ),

            "forward_share_mean":
                _mean(
                    records,
                    "forward_share"
                ),

            "backward_share_mean":
                _mean(
                    records,
                    "backward_share"
                ),

            "directional_jaccard_mean":
                _mean(
                    records,
                    "directional_jaccard"
                ),

            "directional_jaccard_median":
                _median(
                    records,
                    "directional_jaccard"
                ),

            "size_difference_mean":
                _mean(
                    records,
                    "size_difference"
                ),

            "size_ratio_mean":
                _mean(
                    records,
                    "size_ratio"
                )

        }

    return summaries


def print_context_efficiency(
    summaries
):

    print()

    print(
        "=" * 105
    )

    print(
        "DIRECTIONAL CONTEXT EFFICIENCY"
    )

    print(
        "=" * 105
    )

    print()

    print(

        f"{'Metric':<35}"
        f"{'Both correct':>17}"
        f"{'Both wrong':>17}"
        f"{'Forward correct':>19}"
        f"{'Backward correct':>20}"

    )

    print(
        "-" * 110
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

            f"{name:<35}"
            f"{values[0]:>17}"
            f"{values[1]:>17}"
            f"{values[2]:>19}"
            f"{values[3]:>20}"

        )

    row(
        "Forward slice",
        "forward_slice_mean"
    )

    row(
        "Backward slice",
        "backward_slice_mean"
    )

    print()

    row(
        "Forward-only",
        "forward_only_mean"
    )

    row(
        "Backward-only",
        "backward_only_mean"
    )

    print()

    row(
        "Forward context share",
        "forward_share_mean",
        percent=True
    )

    row(
        "Backward context share",
        "backward_share_mean",
        percent=True
    )

    print()

    row(
        "Directional Jaccard",
        "directional_jaccard_mean",
        percent=True
    )

    row(
        "Directional Jaccard median",
        "directional_jaccard_median",
        percent=True
    )

    print()

    row(
        "Forward - backward size",
        "size_difference_mean"
    )

    row(
        "Forward / backward size",
        "size_ratio_mean"
    )

def print_context_efficiency_extremes(
    profiles,
    limit=20
):

    records = sorted(

        profiles,

        key=lambda x:
            abs(
                x["size_difference"]
            ),

        reverse=True

    )

    print()

    print(
        "=" * 100
    )

    print(
        "EXTREME DIRECTIONAL SIZE DIFFERENCES"
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
            "Sample:",
            record["sample_id"]
        )

        print(
            "Forward:",
            record["forward_slice"]
        )

        print(
            "Backward:",
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
            "Directional Jaccard:",
            f"{record['directional_jaccard']:.4f}"
        )

        print(
            "Size difference:",
            record["size_difference"]
        )

def check_duplicate_directional_results(
    directional_results
):

    seen = set()
    duplicates = []

    for result in directional_results:

        sample_id = result.get(
            "sample_id"
        )

        if sample_id in seen:

            duplicates.append(
                sample_id
            )

        else:

            seen.add(
                sample_id
            )

    print()
    print("=" * 80)
    print("DIRECTIONAL RESULT DUPLICATE CHECK")
    print("=" * 80)

    print(
        "Total results:",
        len(directional_results)
    )

    print(
        "Unique sample IDs:",
        len(seen)
    )

    print(
        "Duplicate results:",
        len(duplicates)
    )

    for sample_id in duplicates[:20]:

        print(
            sample_id
        )

    return duplicates