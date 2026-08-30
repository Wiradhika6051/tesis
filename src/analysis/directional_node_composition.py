from collections import Counter, defaultdict
import statistics


def _node_type_map(sample):

    return {

        node_index: node.node_type

        for node_index, node
        in enumerate(sample.cfg["nodes"])

    }


def _get_node_types(
    sample,
    node_ids
):

    node_type_map = _node_type_map(
        sample
    )

    return [

        node_type_map[node_id]

        for node_id in node_ids

        if node_id in node_type_map

    ]


def _composition(
    node_types
):

    counter = Counter(
        node_types
    )

    total = len(
        node_types
    )

    if total == 0:

        return {}

    return {

        node_type:
            count / total

        for node_type, count
        in counter.items()

    }


def _analyze_sample(
    forward_sample,
    backward_sample,
    outcome
):

    forward_nodes = set(
        forward_sample.pruned_cfg["nodes"]
    )

    backward_nodes = set(
        backward_sample.pruned_cfg["nodes"]
    )

    #
    # The pruned CFG nodes themselves may not be
    # directly comparable as objects, so use their
    # node indices.
    #
    forward_ids = set(
        range(
            len(
                forward_sample.pruned_cfg[
                    "nodes"
                ]
            )
        )
    )

    backward_ids = set(
        range(
            len(
                backward_sample.pruned_cfg[
                    "nodes"
                ]
            )
        )
    )

    forward_only_ids = (
        forward_ids
        -
        backward_ids
    )

    backward_only_ids = (
        backward_ids
        -
        forward_ids
    )

    forward_types = _get_node_types(
        forward_sample,
        forward_only_ids
    )

    backward_types = _get_node_types(
        backward_sample,
        backward_only_ids
    )

    return {

        "sample_id":
            (
                forward_sample.repo,
                forward_sample.parent_commit,
                forward_sample.file_path,
                forward_sample.label
            ),

        "outcome":
            outcome,

        "forward_only_count":
            len(forward_types),

        "backward_only_count":
            len(backward_types),

        "forward_only_types":
            Counter(forward_types),

        "backward_only_types":
            Counter(backward_types),

        "forward_only_composition":
            _composition(
                forward_types
            ),

        "backward_only_composition":
            _composition(
                backward_types
            )

    }


def _sample_id(
    sample
):

    return (

        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label

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

        results.append(

            _analyze_sample(

                forward_sample,

                backward_sample,

                comparison[
                    "outcome"
                ]

            )

        )

    return results


def _aggregate_composition(
    records,
    field
):

    total_counter = Counter()
    total_nodes = 0

    for record in records:

        counter = record[field]

        total_counter.update(
            counter
        )

        total_nodes += sum(
            counter.values()
        )

    if total_nodes == 0:

        return {}

    return {

        node_type:
            count / total_nodes

        for node_type, count
        in total_counter.items()

    }


def _aggregate_size(
    records,
    field
):

    values = [

        record[field]

        for record in records

    ]

    if not values:

        return 0.0

    return statistics.mean(
        values
    )


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
            f"DIRECTIONAL NODE COMPOSITION: "
            f"{outcome}"
        )
        print("=" * 80)

        print(
            "Samples:",
            len(records)
        )

        if not records:
            continue

        print()

        print(
            f"Average forward-only nodes : "
            f"{_aggregate_size(
                records,
                "forward_only_count"
            ):.2f}"
        )

        print(
            f"Average backward-only nodes: "
            f"{_aggregate_size(
                records,
                "backward_only_count"
            ):.2f}"
        )

        #
        # Forward-only composition.
        #

        forward_composition = (
            _aggregate_composition(

                records,

                "forward_only_types"

            )
        )

        print()
        print(
            "FORWARD-ONLY NODE TYPES"
        )
        print("-" * 60)

        for node_type, ratio in sorted(

            forward_composition.items(),

            key=lambda x: x[1],

            reverse=True

        )[:top_n]:

            print(

                f"{node_type:<25}"
                f"{ratio * 100:>7.2f}%"

            )

        #
        # Backward-only composition.
        #

        backward_composition = (
            _aggregate_composition(

                records,

                "backward_only_types"

            )
        )

        print()
        print(
            "BACKWARD-ONLY NODE TYPES"
        )
        print("-" * 60)

        for node_type, ratio in sorted(

            backward_composition.items(),

            key=lambda x: x[1],

            reverse=True

        )[:top_n]:

            print(

                f"{node_type:<25}"
                f"{ratio * 100:>7.2f}%"

            )


def print_directional_composition_comparison(
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

    forward_correct = grouped.get(
        "FORWARD_CORRECT",
        []
    )

    backward_correct = grouped.get(
        "BACKWARD_CORRECT",
        []
    )

    #
    # Collect all node types appearing in either
    # group.
    #

    forward_composition = (
        _aggregate_composition(

            forward_correct,

            "forward_only_types"

        )
    )

    backward_composition = (
        _aggregate_composition(

            backward_correct,

            "forward_only_types"

        )
    )

    all_types = set(
        forward_composition
    ) | set(
        backward_composition
    )

    ranking = sorted(

        all_types,

        key=lambda node_type:
            abs(

                forward_composition.get(
                    node_type,
                    0.0
                )
                -
                backward_composition.get(
                    node_type,
                    0.0
                )

            ),

        reverse=True

    )

    print()
    print("=" * 80)
    print(
        "FORWARD-ONLY NODE COMPOSITION"
    )
    print(
        "FORWARD-CORRECT VS BACKWARD-CORRECT"
    )
    print("=" * 80)

    print()

    print(
        f"{'Node Type':<25}"
        f"{'Forward correct':>20}"
        f"{'Backward correct':>20}"
        f"{'Difference':>15}"
    )

    print("-" * 80)

    for node_type in ranking[:top_n]:

        forward_ratio = (
            forward_composition.get(
                node_type,
                0.0
            )
        )

        backward_ratio = (
            backward_composition.get(
                node_type,
                0.0
            )
        )

        difference = (
            forward_ratio
            -
            backward_ratio
        )

        print(

            f"{node_type:<25}"
            f"{forward_ratio * 100:>19.2f}%"
            f"{backward_ratio * 100:>19.2f}%"
            f"{difference * 100:>14.2f}%"

        )

    #
    # --------------------------------------------------
    # Backward-only comparison.
    # --------------------------------------------------
    #

    forward_composition = (
        _aggregate_composition(

            forward_correct,

            "backward_only_types"

        )
    )

    backward_composition = (
        _aggregate_composition(

            backward_correct,

            "backward_only_types"

        )
    )

    all_types = set(
        forward_composition
    ) | set(
        backward_composition
    )

    ranking = sorted(

        all_types,

        key=lambda node_type:
            abs(

                forward_composition.get(
                    node_type,
                    0.0
                )
                -
                backward_composition.get(
                    node_type,
                    0.0
                )

            ),

        reverse=True

    )

    print()
    print("=" * 80)
    print(
        "BACKWARD-ONLY NODE COMPOSITION"
    )
    print(
        "FORWARD-CORRECT VS BACKWARD-CORRECT"
    )
    print("=" * 80)

    print()

    print(
        f"{'Node Type':<25}"
        f"{'Forward correct':>20}"
        f"{'Backward correct':>20}"
        f"{'Difference':>15}"
    )

    print("-" * 80)

    for node_type in ranking[:top_n]:

        forward_ratio = (
            forward_composition.get(
                node_type,
                0.0
            )
        )

        backward_ratio = (
            backward_composition.get(
                node_type,
                0.0
            )
        )

        difference = (
            forward_ratio
            -
            backward_ratio
        )

        print(

            f"{node_type:<25}"
            f"{forward_ratio * 100:>19.2f}%"
            f"{backward_ratio * 100:>19.2f}%"
            f"{difference * 100:>14.2f}%"

        )