from collections import Counter, defaultdict
import statistics

from src.analysis.paired_slice_similarity import (
    get_slice_signatures
)


def _sample_id(sample):

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label
    )


def _get_slice_sets(sample):

    forward = get_slice_signatures(
        sample,
        forward=True
    )

    backward = get_slice_signatures(
        sample,
        forward=False
    )

    return forward, backward


def _extract_node_type(
    signature
):

    return signature[0]

def _composition(signatures):

    if not signatures:

        return Counter()

    return Counter(

        _extract_node_type(signature)

        for signature in signatures

    )


def analyze_directional_node_composition(
    forward_samples,
    backward_samples,
    comparison_results
):

    forward_lookup = {

        _sample_id(sample):
            sample

        for sample in forward_samples

    }

    backward_lookup = {

        _sample_id(sample):
            sample

        for sample in backward_samples

    }

    results = []

    for comparison in comparison_results:

        sample_id = comparison[
            "sample_id"
        ]

        forward_sample = (
            forward_lookup.get(
                sample_id
            )
        )

        backward_sample = (
            backward_lookup.get(
                sample_id
            )
        )

        if (
            forward_sample is None
            or
            backward_sample is None
        ):

            continue

        #
        # Get actual directional slices.
        #
        forward_signatures = (
            get_slice_signatures(
                forward_sample,
                forward=True
            )
        )

        backward_signatures = (
            get_slice_signatures(
                backward_sample,
                forward=False
            )
        )

        #
        # Directional differences.
        #
        forward_only = (

            forward_signatures
            -
            backward_signatures

        )

        backward_only = (

            backward_signatures
            -
            forward_signatures

        )

        overlap = (

            forward_signatures
            &
            backward_signatures

        )

        #
        # Node-type composition.
        #
        forward_types = _composition(
            forward_only
        )

        backward_types = _composition(
            backward_only
        )

        results.append({

            "sample_id":
                sample_id,

            "outcome":
                comparison[
                    "outcome"
                ],

            "label":
                comparison[
                    "label"
                ],

            "forward_only":
                forward_only,

            "backward_only":
                backward_only,

            "overlap":
                overlap,

            "forward_only":
                len(
                    forward_only
                ),

            "backward_only":
                len(
                    backward_only
                ),

            "overlap_count":
                len(
                    overlap
                ),

            "forward_only_types":
                forward_types,

            "backward_only_types":
                backward_types

        })

    return results

def _aggregate_types(
    records,
    field
):

    counter = Counter()

    for record in records:

        counter.update(
            record[field]
        )

    return counter


def print_directional_node_composition(
    results,
    top_n=15
):

    grouped = defaultdict(list)

    for record in results:

        grouped[
            record["outcome"]
        ].append(
            record
        )

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
        print("=" * 80)
        print(
            f"DIRECTIONAL NODE COMPOSITION"
        )
        print(
            f"Outcome: {outcome}"
        )
        print("=" * 80)

        print(
            "Samples:",
            len(records)
        )

        if not records:
            continue

        avg_forward = statistics.mean(

            record[
                "forward_only"
            ]

            for record in records

        )

        avg_backward = statistics.mean(

            record[
                "backward_only"
            ]

            for record in records

        )

        avg_overlap = statistics.mean(

            record[
                "overlap_count"
            ]

            for record in records

        )

        print()
        print(
            f"Average forward-only  : "
            f"{avg_forward:.2f}"
        )

        print(
            f"Average backward-only : "
            f"{avg_backward:.2f}"
        )

        print(
            f"Average overlap       : "
            f"{avg_overlap:.2f}"
        )

        #
        # Forward-only.
        #
        forward_counter = _aggregate_types(

            records,

            "forward_only_types"

        )

        forward_total = sum(
            forward_counter.values()
        )

        print()
        print(
            "FORWARD-ONLY NODE TYPES"
        )
        print("-" * 60)

        for node_type, count in (
            forward_counter.most_common(
                top_n
            )
        ):

            ratio = (

                count
                /
                forward_total

                if forward_total
                else 0

            )

            print(

                f"{node_type:<25}"
                f"{count:>8}"
                f" ({ratio * 100:>6.2f}%)"

            )

        #
        # Backward-only.
        #
        backward_counter = _aggregate_types(

            records,

            "backward_only_types"

        )

        backward_total = sum(
            backward_counter.values()
        )

        print()
        print(
            "BACKWARD-ONLY NODE TYPES"
        )
        print("-" * 60)

        for node_type, count in (
            backward_counter.most_common(
                top_n
            )
        ):

            ratio = (

                count
                /
                backward_total

                if backward_total
                else 0

            )

            print(

                f"{node_type:<25}"
                f"{count:>8}"
                f" ({ratio * 100:>6.2f}%)"

            )