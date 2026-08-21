from collections import Counter, defaultdict


def analyze_directional_slice_composition(
    samples
):
    """
    Analyze the structural composition of
    forward and backward slices.

    Requires:
        sample.cfg
        sample.seed_nodes
        sample.function_nodes
    """

    overall = {

        "samples": 0,

        "forward_only_nodes": 0,
        "backward_only_nodes": 0,
        "overlap_nodes": 0,

        "forward_only_types": Counter(),
        "backward_only_types": Counter(),
        "overlap_types": Counter(),

        "by_label": defaultdict(
            lambda: {

                "samples": 0,

                "forward_only_nodes": 0,
                "backward_only_nodes": 0,
                "overlap_nodes": 0,

                "forward_only_types": Counter(),
                "backward_only_types": Counter(),
                "overlap_types": Counter(),

            }
        )
    }

    for sample in samples:

        if sample.cfg is None:
            continue

        if not sample.seed_nodes:
            continue

        if not sample.function_nodes:
            continue

        cfg = sample.cfg

        seed_nodes = set(
            sample.seed_nodes
        )

        function_nodes = set(
            sample.function_nodes
        )

        #
        # Calculate slices.
        #
        backward_nodes = get_slice_nodes(
            cfg,
            seed_nodes,
            function_nodes,
            forward=False
        )

        forward_nodes = get_slice_nodes(
            cfg,
            seed_nodes,
            function_nodes,
            forward=True
        )

        #
        # Convert to sets.
        #
        backward_nodes = set(
            backward_nodes
        )

        forward_nodes = set(
            forward_nodes
        )

        #
        # Directional components.
        #
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

        #
        # Build node lookup.
        #
        node_lookup = {

            node.node_id: node

            for node in cfg["nodes"]

        }

        #
        # Update overall.
        #
        overall["samples"] += 1

        overall[
            "forward_only_nodes"
        ] += len(forward_only)

        overall[
            "backward_only_nodes"
        ] += len(backward_only)

        overall[
            "overlap_nodes"
        ] += len(overlap)

        #
        # Label statistics.
        #
        label_stats = overall[
            "by_label"
        ][sample.label]

        label_stats[
            "samples"
        ] += 1

        label_stats[
            "forward_only_nodes"
        ] += len(forward_only)

        label_stats[
            "backward_only_nodes"
        ] += len(backward_only)

        label_stats[
            "overlap_nodes"
        ] += len(overlap)

        #
        # Node type statistics.
        #
        for node_id in forward_only:

            node = node_lookup.get(
                node_id
            )

            if node is None:
                continue

            node_type = (
                node.node_type
            )

            overall[
                "forward_only_types"
            ][node_type] += 1

            label_stats[
                "forward_only_types"
            ][node_type] += 1

        for node_id in backward_only:

            node = node_lookup.get(
                node_id
            )

            if node is None:
                continue

            node_type = (
                node.node_type
            )

            overall[
                "backward_only_types"
            ][node_type] += 1

            label_stats[
                "backward_only_types"
            ][node_type] += 1

        for node_id in overlap:

            node = node_lookup.get(
                node_id
            )

            if node is None:
                continue

            node_type = (
                node.node_type
            )

            overall[
                "overlap_types"
            ][node_type] += 1

            label_stats[
                "overlap_types"
            ][node_type] += 1

    return overall

def print_directional_slice_composition(
    results,
    top_n=15
):

    print()
    print("=" * 80)
    print(
        "DIRECTIONAL SLICE COMPOSITION ANALYSIS"
    )
    print("=" * 80)

    samples = results["samples"]

    if samples == 0:

        print(
            "No valid samples."
        )

        return

    print()

    print(
        f"Samples analyzed : {samples}"
    )

    print()

    avg_forward_only = (
        results["forward_only_nodes"]
        /
        samples
    )

    avg_backward_only = (
        results["backward_only_nodes"]
        /
        samples
    )

    avg_overlap = (
        results["overlap_nodes"]
        /
        samples
    )

    print("## Average Slice Components")

    print()

    print(
        f"Forward only : "
        f"{avg_forward_only:.2f}"
    )

    print(
        f"Backward only: "
        f"{avg_backward_only:.2f}"
    )

    print(
        f"Overlap      : "
        f"{avg_overlap:.2f}"
    )

    print()

    print("=" * 80)
    print(
        "FORWARD-ONLY NODE TYPES"
    )
    print("=" * 80)

    for node_type, count in (
        results[
            "forward_only_types"
        ].most_common(top_n)
    ):

        print(
            f"{node_type:<25} {count}"
        )

    print()

    print("=" * 80)
    print(
        "BACKWARD-ONLY NODE TYPES"
    )
    print("=" * 80)

    for node_type, count in (
        results[
            "backward_only_types"
        ].most_common(top_n)
    ):

        print(
            f"{node_type:<25} {count}"
        )

    print()

    print("=" * 80)
    print(
        "OVERLAPPING NODE TYPES"
    )
    print("=" * 80)

    for node_type, count in (
        results[
            "overlap_types"
        ].most_common(top_n)
    ):

        print(
            f"{node_type:<25} {count}"
        )

    print()

    print("=" * 80)
    print(
        "COMPOSITION BY LABEL"
    )
    print("=" * 80)

    for label, stats in sorted(
        results["by_label"].items()
    ):

        samples = stats[
            "samples"
        ]

        print()

        print(
            f"Label {label}"
        )

        print(
            "-" * 50
        )

        print(
            f"Samples: {samples}"
        )

        print(
            f"Avg forward-only: "
            f"{stats['forward_only_nodes'] / samples:.2f}"
        )

        print(
            f"Avg backward-only: "
            f"{stats['backward_only_nodes'] / samples:.2f}"
        )

        print(
            f"Avg overlap: "
            f"{stats['overlap_nodes'] / samples:.2f}"
        )

        print()

        print(
            "Top forward-only types:"
        )

        for node_type, count in (
            stats[
                "forward_only_types"
            ].most_common(top_n)
        ):

            print(
                f"  {node_type:<22} {count}"
            )

        print()

        print(
            "Top backward-only types:"
        )

        for node_type, count in (
            stats[
                "backward_only_types"
            ].most_common(top_n)
        ):

            print(
                f"  {node_type:<22} {count}"
            )