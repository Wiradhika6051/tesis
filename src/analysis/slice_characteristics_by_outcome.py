from collections import defaultdict, Counter
import statistics

from src.analysis.slice_analysis import get_slice_nodes


def get_sample_id(sample):

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label
    )


def build_sample_lookup(samples):

    lookup = {}

    for sample in samples:

        lookup[
            get_sample_id(sample)
        ] = sample

    return lookup


def get_directional_slice(
    sample,
    forward
):

    return get_slice_nodes(

        sample.cfg,

        set(sample.seed_nodes),

        set(sample.function_nodes),

        forward=forward

    )


def calculate_characteristics(
    sample
):

    function_nodes = set(
        sample.function_nodes
    )

    seed_nodes = set(
        sample.seed_nodes
    )

    forward_nodes = get_directional_slice(
        sample,
        forward=True
    )

    backward_nodes = get_directional_slice(
        sample,
        forward=False
    )

    overlap = (
        forward_nodes
        &
        backward_nodes
    )

    forward_only = (
        forward_nodes
        -
        backward_nodes
    )

    backward_only = (
        backward_nodes
        -
        forward_nodes
    )

    function_size = len(
        function_nodes
    )

    return {

        "forward_size":
            len(forward_nodes),

        "backward_size":
            len(backward_nodes),

        "overlap_size":
            len(overlap),

        "forward_only_size":
            len(forward_only),

        "backward_only_size":
            len(backward_only),

        "forward_ratio":
            (
                len(forward_nodes)
                /
                function_size
            )
            if function_size
            else 0.0,

        "backward_ratio":
            (
                len(backward_nodes)
                /
                function_size
            )
            if function_size
            else 0.0,

        "overlap_ratio":
            (
                len(overlap)
                /
                len(
                    forward_nodes
                    |
                    backward_nodes
                )
            )
            if (
                forward_nodes
                |
                backward_nodes
            )
            else 0.0,

        "seed_count":
            len(seed_nodes),

        "function_size":
            function_size
    }


def analyze_slice_characteristics_by_prediction_outcome(
    comparison_results,
    forward_samples,
    backward_samples
):

    forward_lookup = build_sample_lookup(
        forward_samples
    )

    backward_lookup = build_sample_lookup(
        backward_samples
    )

    grouped = defaultdict(list)

    #
    # comparison_results should be the result
    # returned by compare_forward_backward_predictions()
    #
    records = comparison_results.get(
        "records",
        comparison_results.get(
            "predictions",
            []
        )
    )

    for record in records:

        sample_id = record.get(
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
            continue

        label = record["label"]

        forward_prediction = (
            record["forward_prediction"]
        )

        backward_prediction = (
            record["backward_prediction"]
        )

        forward_correct = (
            forward_prediction == label
        )

        backward_correct = (
            backward_prediction == label
        )

        if (
            forward_correct
            and
            backward_correct
        ):

            outcome = "BOTH_CORRECT"

        elif (
            not forward_correct
            and
            not backward_correct
        ):

            outcome = "BOTH_WRONG"

        elif forward_correct:

            outcome = "FORWARD_CORRECT"

        elif backward_correct:

            outcome = "BACKWARD_CORRECT"

        else:

            outcome = "UNKNOWN"

        forward_characteristics = (
            calculate_characteristics(
                forward_sample
            )
        )

        backward_characteristics = (
            calculate_characteristics(
                backward_sample
            )
        )

        grouped[outcome].append({

            "sample_id":
                sample_id,

            "label":
                label,

            "forward_prediction":
                forward_prediction,

            "backward_prediction":
                backward_prediction,

            "forward":
                forward_characteristics,

            "backward":
                backward_characteristics

        })

    return {

        "grouped":
            grouped

    }


def average(
    records,
    direction,
    key
):

    values = [

        record[direction][key]

        for record in records

    ]

    if not values:
        return 0.0

    return statistics.mean(
        values
    )


def print_slice_characteristics_by_prediction_outcome(
    analysis
):

    grouped = analysis[
        "grouped"
    ]

    outcomes = [

        "BOTH_CORRECT",

        "BOTH_WRONG",

        "FORWARD_CORRECT",

        "BACKWARD_CORRECT"

    ]

    print()

    print(
        "=" * 80
    )

    print(
        "SLICE CHARACTERISTICS BY PREDICTION OUTCOME"
    )

    print(
        "=" * 80
    )

    for outcome in outcomes:

        records = grouped.get(
            outcome,
            []
        )

        print()

        print(
            outcome
        )

        print(
            "-" * 60
        )

        print(
            "Samples:",
            len(records)
        )

        if not records:
            continue

        print()

        print(
            "                         Forward      Backward"
        )

        print(
            f"Average slice size     : "
            f"{average(records, 'forward', 'forward_size'):>10.2f} "
            f"{average(records, 'backward', 'backward_size'):>12.2f}"
        )

        print(
            f"Average retention      : "
            f"{average(records, 'forward', 'forward_ratio'):>10.2%} "
            f"{average(records, 'backward', 'backward_ratio'):>12.2%}"
        )

        print(
            f"Average overlap        : "
            f"{average(records, 'forward', 'overlap_size'):>10.2f} "
            f"{average(records, 'backward', 'overlap_size'):>12.2f}"
        )

        print(
            f"Average forward-only   : "
            f"{average(records, 'forward', 'forward_only_size'):>10.2f}"
        )

        print(
            f"Average backward-only  : "
            f"{average(records, 'forward', 'backward_only_size'):>10.2f}"
        )

        print(
            f"Average seed count     : "
            f"{statistics.mean([r['forward']['seed_count'] for r in records]):.2f}"
        )

        print(
            f"Average function size  : "
            f"{statistics.mean([r['forward']['function_size'] for r in records]):.2f}"
        )

from collections import defaultdict
import statistics


def analyze_slice_characteristics_by_outcome(
    comparison_results,
    forward_samples,
    backward_samples
):

    #
    # Build sample lookup.
    #
    forward_lookup = {
        sample.graph.sample_id: sample
        for sample in forward_samples
        if sample.graph is not None
    }

    backward_lookup = {
        sample.graph.sample_id: sample
        for sample in backward_samples
        if sample.graph is not None
    }

    grouped = defaultdict(list)

    comparisons = comparison_results[
        "comparisons"
    ]

    for comparison in comparisons:

        sample_id = comparison[
            "sample_id"
        ]

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
            continue

        #
        # Slice sizes.
        #
        forward_nodes = len(
            forward_sample.pruned_cfg["nodes"]
        )

        backward_nodes = len(
            backward_sample.pruned_cfg["nodes"]
        )

        #
        # Function size.
        #
        function_nodes = len(
            forward_sample.function_nodes
        )

        #
        # Number of seed nodes.
        #
        seeds = len(
            forward_sample.seed_nodes
        )

        #
        # Store characteristics.
        #
        record = {

            "sample_id":
                sample_id,

            "label":
                comparison["label"],

            "forward_prediction":
                comparison["forward_prediction"],

            "backward_prediction":
                comparison["backward_prediction"],

            "comparison":
                comparison["comparison"],

            "function_nodes":
                function_nodes,

            "seed_nodes":
                seeds,

            "forward_nodes":
                forward_nodes,

            "backward_nodes":
                backward_nodes,

            "forward_ratio":
                (
                    forward_nodes / function_nodes
                    if function_nodes > 0
                    else 0
                ),

            "backward_ratio":
                (
                    backward_nodes / function_nodes
                    if function_nodes > 0
                    else 0
                )

        }

        grouped[
            comparison["comparison"]
        ].append(
            record
        )

    return grouped

def print_slice_characteristics_by_outcome(
    grouped
):

    print()
    print("=" * 80)
    print("SLICE CHARACTERISTICS BY PREDICTION OUTCOME")
    print("=" * 80)

    outcomes = [

        "BOTH_CORRECT",
        "BOTH_WRONG",
        "FORWARD_CORRECT",
        "BACKWARD_CORRECT"

    ]

    for outcome in outcomes:

        records = grouped.get(
            outcome,
            []
        )

        print()
        print(outcome)
        print("-" * 60)

        print(
            f"Samples: {len(records)}"
        )

        if not records:
            continue

        function_sizes = [
            r["function_nodes"]
            for r in records
        ]

        seed_sizes = [
            r["seed_nodes"]
            for r in records
        ]

        forward_sizes = [
            r["forward_nodes"]
            for r in records
        ]

        backward_sizes = [
            r["backward_nodes"]
            for r in records
        ]

        forward_ratios = [
            r["forward_ratio"]
            for r in records
        ]

        backward_ratios = [
            r["backward_ratio"]
            for r in records
        ]

        print(
            f"Average function nodes : "
            f"{statistics.mean(function_sizes):.2f}"
        )

        print(
            f"Average seed nodes     : "
            f"{statistics.mean(seed_sizes):.2f}"
        )

        print(
            f"Average forward nodes  : "
            f"{statistics.mean(forward_sizes):.2f}"
        )

        print(
            f"Average backward nodes : "
            f"{statistics.mean(backward_sizes):.2f}"
        )

        print(
            f"Average forward ratio  : "
            f"{statistics.mean(forward_ratios):.2%}"
        )

        print(
            f"Average backward ratio : "
            f"{statistics.mean(backward_ratios):.2%}"
        )