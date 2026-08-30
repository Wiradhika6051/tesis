from collections import defaultdict
import statistics


def _distance_percentage(
    buckets,
    total
):

    if total == 0:

        return {
            "1": 0.0,
            "2-3": 0.0,
            "4-5": 0.0,
            ">5": 0.0
        }

    return {

        bucket:
            buckets[bucket] / total

        for bucket in [
            "1",
            "2-3",
            "4-5",
            ">5"
        ]

    }


def _summarize_group(
    records,
    field
):

    if not records:
        return None

    values = [

        record[field]

        for record in records

    ]

    return {

        "average":
            statistics.mean(values),

        "median":
            statistics.median(values),

        "minimum":
            min(values),

        "maximum":
            max(values)

    }


def analyze_normalized_distance(
    distance_results
):

    grouped = defaultdict(list)

    #
    # Convert the existing distance analysis
    # into per-sample normalized statistics.
    #
    for record in distance_results:

        forward_buckets = (
            record[
                "forward_only_buckets"
            ]
        )

        backward_buckets = (
            record[
                "backward_only_buckets"
            ]
        )

        forward_total = sum(
            forward_buckets.values()
        )

        backward_total = sum(
            backward_buckets.values()
        )

        forward_percentages = (
            _distance_percentage(
                forward_buckets,
                forward_total
            )
        )

        backward_percentages = (
            _distance_percentage(
                backward_buckets,
                backward_total
            )
        )

        #
        # Distant context = >5 hops.
        #
        forward_distant = (
            forward_percentages[">5"]
        )

        backward_distant = (
            backward_percentages[">5"]
        )

        #
        # Local context = distance 1-3.
        #
        forward_local = (
            forward_percentages["1"]
            +
            forward_percentages["2-3"]
        )

        backward_local = (
            backward_percentages["1"]
            +
            backward_percentages["2-3"]
        )

        result = {

            "sample_id":
                record["sample_id"],

            "outcome":
                record["outcome"],

            "label":
                record["label"],

            "forward_size":
                record["forward_size"],

            "backward_size":
                record["backward_size"],

            "forward_only":
                record["forward_only"],

            "backward_only":
                record["backward_only"],

            "seed_count":
                record["seed_count"],

            #
            # Normalized distance distribution.
            #
            "forward_distance_1":
                forward_percentages["1"],

            "forward_distance_2_3":
                forward_percentages["2-3"],

            "forward_distance_4_5":
                forward_percentages["4-5"],

            "forward_distance_gt5":
                forward_percentages[">5"],

            "backward_distance_1":
                backward_percentages["1"],

            "backward_distance_2_3":
                backward_percentages["2-3"],

            "backward_distance_4_5":
                backward_percentages["4-5"],

            "backward_distance_gt5":
                backward_percentages[">5"],

            #
            # Aggregated local/distant context.
            #
            "forward_local_ratio":
                forward_local,

            "backward_local_ratio":
                backward_local,

            "forward_distant_ratio":
                forward_distant,

            "backward_distant_ratio":
                backward_distant

        }

        grouped[
            record["outcome"]
        ].append(
            result
        )

    return grouped


def _print_metric(
    records,
    field
):

    values = [

        record[field]

        for record in records

    ]

    if not values:

        print(
            "N/A"
        )

        return

    print(
        f"Average : "
        f"{statistics.mean(values) * 100:.2f}%"
    )

    print(
        f"Median  : "
        f"{statistics.median(values) * 100:.2f}%"
    )


def print_normalized_distance_analysis(
    grouped
):

    outcomes = [

        "BOTH_CORRECT",
        "BOTH_WRONG",
        "FORWARD_CORRECT",
        "BACKWARD_CORRECT"

    ]

    print()
    print("=" * 80)
    print(
        "NORMALIZED CFG DISTANCE BY PREDICTION OUTCOME"
    )
    print("=" * 80)

    for outcome in outcomes:

        records = grouped.get(
            outcome,
            []
        )

        print()
        print(outcome)
        print("-" * 60)

        print(
            "Samples:",
            len(records)
        )

        if not records:
            continue

        print()

        print(
            "FORWARD-ONLY DISTANCE DISTRIBUTION"
        )

        print(
            "Distance 1:",
            end=" "
        )

        _print_metric(
            records,
            "forward_distance_1"
        )

        print(
            "Distance 2-3:",
            end=" "
        )

        _print_metric(
            records,
            "forward_distance_2_3"
        )

        print(
            "Distance 4-5:",
            end=" "
        )

        _print_metric(
            records,
            "forward_distance_4_5"
        )

        print(
            "Distance >5:",
            end=" "
        )

        _print_metric(
            records,
            "forward_distance_gt5"
        )

        print()

        print(
            "BACKWARD-ONLY DISTANCE DISTRIBUTION"
        )

        print(
            "Distance 1:",
            end=" "
        )

        _print_metric(
            records,
            "backward_distance_1"
        )

        print(
            "Distance 2-3:",
            end=" "
        )

        _print_metric(
            records,
            "backward_distance_2_3"
        )

        print(
            "Distance 4-5:",
            end=" "
        )

        _print_metric(
            records,
            "backward_distance_4_5"
        )

        print(
            "Distance >5:",
            end=" "
        )

        _print_metric(
            records,
            "backward_distance_gt5"
        )

        print()

        print(
            "AGGREGATED CONTEXT"
        )

        print(
            "Forward local (1-3):",
            end=" "
        )

        _print_metric(
            records,
            "forward_local_ratio"
        )

        print(
            "Forward distant (>5):",
            end=" "
        )

        _print_metric(
            records,
            "forward_distant_ratio"
        )

        print(
            "Backward local (1-3):",
            end=" "
        )

        _print_metric(
            records,
            "backward_local_ratio"
        )

        print(
            "Backward distant (>5):",
            end=" "
        )

        _print_metric(
            records,
            "backward_distant_ratio"
        )


def print_distance_comparison(
    grouped
):

    forward_correct = grouped.get(
        "FORWARD_CORRECT",
        []
    )

    backward_correct = grouped.get(
        "BACKWARD_CORRECT",
        []
    )

    both_correct = grouped.get(
        "BOTH_CORRECT",
        []
    )

    both_wrong = grouped.get(
        "BOTH_WRONG",
        []
    )

    print()
    print("=" * 80)
    print(
        "DISTANT CONTEXT COMPARISON"
    )
    print("=" * 80)

    print()

    print(
        f"{'Metric':<35}"
        f"{'Both correct':>18}"
        f"{'Both wrong':>18}"
        f"{'Forward correct':>20}"
        f"{'Backward correct':>21}"
    )

    print("-" * 115)

    metrics = [

        (
            "Forward >5 hops",
            "forward_distant_ratio"
        ),

        (
            "Backward >5 hops",
            "backward_distant_ratio"
        ),

        (
            "Forward local (1-3)",
            "forward_local_ratio"
        ),

        (
            "Backward local (1-3)",
            "backward_local_ratio"
        )

    ]

    groups = [

        both_correct,
        both_wrong,
        forward_correct,
        backward_correct

    ]

    for name, field in metrics:

        values = []

        for group in groups:

            if not group:

                values.append(
                    "N/A"
                )

                continue

            value = statistics.mean(
                record[field]
                for record in group
            )

            values.append(
                f"{value * 100:.2f}%"
            )

        print(
            f"{name:<35}"
            f"{values[0]:>18}"
            f"{values[1]:>18}"
            f"{values[2]:>20}"
            f"{values[3]:>21}"
        )