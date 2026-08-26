from collections import defaultdict
from collections import Counter

import statistics


def get_sample_id(
    sample
):

    return (

        sample.repo,

        sample.commit_hash,

        sample.file_path,

        sample.label

    )


def get_slice_nodes(
    cfg,
    seed_nodes,
    function_nodes,
    forward
):

    """
    Use your existing slice traversal implementation here.

    This function is expected to return a set
    containing the node IDs in the slice.
    """

    #
    # Replace this import if get_slice_nodes
    # already lives somewhere else.
    #
    from src.analysis.slice_analysis import (
        get_slice_nodes as existing_get_slice_nodes
    )

    return existing_get_slice_nodes(

        cfg,

        seed_nodes,

        function_nodes,

        forward=forward

    )


def analyze_directional_slice_composition(

    samples,

    directional_analysis

):

    #
    # --------------------------------------------------
    # Build category lookup.
    # --------------------------------------------------
    #

    category_lookup = {

        result[
            "sample_id"
        ]:

        result[
            "category"
        ]

        for result in directional_analysis[
            "results"
        ]

    }

    #
    # --------------------------------------------------
    # Storage.
    # --------------------------------------------------
    #

    grouped = defaultdict(
        list
    )

    node_types = defaultdict(
        lambda: {

            "forward_only":
                Counter(),

            "backward_only":
                Counter(),

            "overlap":
                Counter()

        }
    )

    missing_samples = []

    #
    # --------------------------------------------------
    # Analyze each sample.
    # --------------------------------------------------
    #

    for sample in samples:

        sample_id = get_sample_id(
            sample
        )

        category = (
            category_lookup.get(
                sample_id
            )
        )

        #
        # Sample not present in directional comparison.
        #
        if category is None:

            missing_samples.append(
                sample_id
            )

            continue

        seed_nodes = set(
            sample.seed_nodes
        )

        function_nodes = set(
            sample.function_nodes
        )

        if not seed_nodes:

            continue

        if not function_nodes:

            continue

        #
        # Calculate slices.
        #
        forward_nodes = get_slice_nodes(

            sample.cfg,

            seed_nodes,

            function_nodes,

            forward=True

        )

        backward_nodes = get_slice_nodes(

            sample.cfg,

            seed_nodes,

            function_nodes,

            forward=False

        )

        #
        # Compare composition.
        #
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

        overlap = (

            forward_nodes
            &
            backward_nodes

        )

        #
        # Store numerical information.
        #
        grouped[
            category
        ].append({

            "sample_id":
                sample_id,

            "label":
                sample.label,

            "function_size":
                len(
                    function_nodes
                ),

            "seed_count":
                len(
                    seed_nodes
                ),

            "forward_size":
                len(
                    forward_nodes
                ),

            "backward_size":
                len(
                    backward_nodes
                ),

            "forward_only":
                len(
                    forward_only
                ),

            "backward_only":
                len(
                    backward_only
                ),

            "overlap":
                len(
                    overlap
                )

        })

        #
        # --------------------------------------------------
        # Count node types.
        # --------------------------------------------------
        #

        for node_id in forward_only:

            node = sample.cfg[
                "nodes"
            ][
                node_id
            ]

            node_type = getattr(
                node,
                "node_type",
                None
            )

            if node_type is None:

                node_type = type(
                    node
                ).__name__

            node_types[
                category
            ][
                "forward_only"
            ][
                node_type
            ] += 1

        for node_id in backward_only:

            node = sample.cfg[
                "nodes"
            ][
                node_id
            ]

            node_type = getattr(
                node,
                "node_type",
                None
            )

            if node_type is None:

                node_type = type(
                    node
                ).__name__

            node_types[
                category
            ][
                "backward_only"
            ][
                node_type
            ] += 1

        for node_id in overlap:

            node = sample.cfg[
                "nodes"
            ][
                node_id
            ]

            node_type = getattr(
                node,
                "node_type",
                None
            )

            if node_type is None:

                node_type = type(
                    node
                ).__name__

            node_types[
                category
            ][
                "overlap"
            ][
                node_type
            ] += 1

    return {

        "grouped":
            grouped,

        "node_types":
            node_types,

        "missing_samples":
            missing_samples

    }


def print_directional_slice_composition(
    analysis
):

    print()

    print(
        "=" * 80
    )

    print(
        "DIRECTIONAL SLICE COMPOSITION BY MODEL OUTCOME"
    )

    print(
        "=" * 80
    )

    grouped = analysis[
        "grouped"
    ]

    node_types = analysis[
        "node_types"
    ]

    categories = [

        "both_correct",

        "forward_wins",

        "backward_wins",

        "both_wrong"

    ]

    metrics = [

        "function_size",

        "seed_count",

        "forward_size",

        "backward_size",

        "forward_only",

        "backward_only",

        "overlap"

    ]

    for category in categories:

        records = grouped.get(
            category,
            []
        )

        print()

        print(
            category.upper()
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

        for metric in metrics:

            values = [

                record[
                    metric
                ]

                for record in records

            ]

            print(

                f"Average {metric:20s}: "
                f"{statistics.mean(values):.2f}"

            )

        #
        # Node types.
        #
        print()

        print(
            "Top forward-only node types:"
        )

        for (

            node_type,
            count

        ) in node_types[
            category
        ][
            "forward_only"
        ].most_common(
            10
        ):

            print(

                f"  {node_type:20s} "
                f"{count}"

            )

        print()

        print(
            "Top backward-only node types:"
        )

        for (

            node_type,
            count

        ) in node_types[
            category
        ][
            "backward_only"
        ].most_common(
            10
        ):

            print(

                f"  {node_type:20s} "
                f"{count}"

            )

    print()

    print(
        "Samples missing from directional comparison:",
        len(
            analysis[
                "missing_samples"
            ]
        )
    )