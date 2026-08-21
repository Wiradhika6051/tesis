from collections import Counter, defaultdict

from src.analysis.slice_analysis import get_slice_nodes


def analyze_directional_slice_composition(
    sample
):
    """
    Analyze the composition of:

    - forward-only nodes
    - backward-only nodes
    - overlap nodes

    for a single sample.
    """

    cfg = sample.cfg

    if cfg is None:
        return None

    if not sample.seed_nodes:
        return None

    if not sample.function_nodes:
        return None

    seed_nodes = set(
        sample.seed_nodes
    )

    function_nodes = set(
        sample.function_nodes
    )

    #
    # Calculate slices.
    #
    forward_nodes = get_slice_nodes(
        cfg,
        seed_nodes,
        function_nodes,
        forward=True
    )

    backward_nodes = get_slice_nodes(
        cfg,
        seed_nodes,
        function_nodes,
        forward=False
    )

    #
    # Normalize just in case the function
    # returns a list.
    #
    forward_nodes = set(
        forward_nodes
    )

    backward_nodes = set(
        backward_nodes
    )

    #
    # Calculate directional groups.
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
    # Build node lookup.
    #
    node_lookup = {

        node.node_id: node

        for node in cfg["nodes"]

    }

    #
    # Count node types.
    #
    def count_node_types(
        node_ids
    ):

        counter = Counter()

        for node_id in node_ids:

            node = node_lookup.get(
                node_id
            )

            if node is None:
                continue

            counter[
                node.node_type
            ] += 1

        return counter

    result = {

        "repo":
            sample.repo,

        "file":
            sample.file_path,

        "label":
            sample.label,

        "seed_count":
            len(seed_nodes),

        "forward_count":
            len(forward_nodes),

        "backward_count":
            len(backward_nodes),

        "forward_only_count":
            len(forward_only),

        "backward_only_count":
            len(backward_only),

        "overlap_count":
            len(overlap),

        "forward_only_types":
            count_node_types(
                forward_only
            ),

        "backward_only_types":
            count_node_types(
                backward_only
            ),

        "overlap_types":
            count_node_types(
                overlap
            )

    }

    return result

def summarize_directional_slice_composition(
    samples
):

    print()
    print("=" * 80)
    print("DIRECTIONAL SLICE COMPOSITION ANALYSIS")
    print("=" * 80)

    results = []

    #
    # Global counters.
    #
    overall = {

        "forward_only":
            Counter(),

        "backward_only":
            Counter(),

        "overlap":
            Counter()

    }

    #
    # Counters by label.
    #
    by_label = defaultdict(
        lambda: {

            "forward_only":
                Counter(),

            "backward_only":
                Counter(),

            "overlap":
                Counter()

        }
    )

    #
    # Sample statistics.
    #
    sample_stats = defaultdict(
        Counter
    )

    for sample in samples:

        result = (
            analyze_directional_slice_composition(
                sample
            )
        )

        if result is None:
            continue

        results.append(
            result
        )

        label = result["label"]

        #
        # Aggregate node types.
        #
        for group in [

            "forward_only",
            "backward_only",
            "overlap"

        ]:

            type_key = (
                f"{group}_types"
            )

            overall[group].update(
                result[type_key]
            )

            by_label[label][group].update(
                result[type_key]
            )

        #
        # Aggregate sample counts.
        #
        sample_stats["overall"][
            "samples"
        ] += 1

        sample_stats["overall"][
            "forward_only_nodes"
        ] += result[
            "forward_only_count"
        ]

        sample_stats["overall"][
            "backward_only_nodes"
        ] += result[
            "backward_only_count"
        ]

        sample_stats["overall"][
            "overlap_nodes"
        ] += result[
            "overlap_count"
        ]

        sample_stats[label][
            "samples"
        ] += 1

        sample_stats[label][
            "forward_only_nodes"
        ] += result[
            "forward_only_count"
        ]

        sample_stats[label][
            "backward_only_nodes"
        ] += result[
            "backward_only_count"
        ]

        sample_stats[label][
            "overlap_nodes"
        ] += result[
            "overlap_count"
        ]

    #
    # Print sample statistics.
    #
    print()
    print("SAMPLE-LEVEL STATISTICS")

    for key, stats in sample_stats.items():

        samples_count = stats["samples"]

        if samples_count == 0:
            continue

        print()
        print(
            f"Group: {key}"
        )

        print(
            "Samples:",
            samples_count
        )

        print(
            "Average forward-only nodes:",
            (
                stats["forward_only_nodes"]
                /
                samples_count
            )
        )

        print(
            "Average backward-only nodes:",
            (
                stats["backward_only_nodes"]
                /
                samples_count
            )
        )

        print(
            "Average overlap nodes:",
            (
                stats["overlap_nodes"]
                /
                samples_count
            )
        )

    #
    # Print overall node type distribution.
    #
    print()
    print("=" * 80)
    print("OVERALL NODE TYPE DISTRIBUTION")
    print("=" * 80)

    for group in [

        "forward_only",
        "backward_only",
        "overlap"

    ]:

        print()
        print(
            group
            .upper()
            .replace("_", " ")
        )

        print("-" * 50)

        for node_type, count in (
            overall[group]
            .most_common()
        ):

            print(
                f"{node_type:<25} {count}"
            )

    #
    # Print distributions by label.
    #
    for label in sorted(
        by_label.keys()
    ):

        print()
        print("=" * 80)

        print(
            f"LABEL {label}"
        )

        print("=" * 80)

        for group in [

            "forward_only",
            "backward_only",
            "overlap"

        ]:

            print()
            print(
                group
                .upper()
                .replace("_", " ")
            )

            print("-" * 50)

            for node_type, count in (
                by_label[label][group]
                .most_common()
            ):

                print(
                    f"{node_type:<25} {count}"
                )

    return {

        "results":
            results,

        "overall":
            overall,

        "by_label":
            by_label,

        "sample_stats":
            sample_stats

    }

def inspect_directional_slice(
    sample
):

    analysis = sample.slice_analysis

    forward_ids = set(
        analysis["forward_node_ids"]
    )

    backward_ids = set(
        analysis["backward_node_ids"]
    )

    node_map = {
        node.node_id: node
        for node in sample.cfg["nodes"]
    }

    overlap = (
        forward_ids
        &
        backward_ids
    )

    forward_only = (
        forward_ids
        -
        backward_ids
    )

    backward_only = (
        backward_ids
        -
        forward_ids
    )

    print("=" * 80)
    print("DIRECTIONAL SLICE INSPECTION")
    print("=" * 80)

    print()

    print(
        "Repo:",
        sample.repo
    )

    print(
        "File:",
        sample.file_path
    )

    print(
        "Label:",
        sample.label
    )

    print()

    print(
        "Seed nodes:",
        analysis["seed_node_ids"]
    )

    print()

    print(
        f"Forward nodes: "
        f"{len(forward_ids)}"
    )

    print(
        f"Backward nodes: "
        f"{len(backward_ids)}"
    )

    print(
        f"Overlap: "
        f"{len(overlap)}"
    )

    print(
        f"Forward only: "
        f"{len(forward_only)}"
    )

    print(
        f"Backward only: "
        f"{len(backward_only)}"
    )

    print()

    print("=" * 80)
    print("SEED NODES")
    print("=" * 80)

    for node_id in sorted(
        analysis["seed_node_ids"]
    ):

        node = node_map.get(
            node_id
        )

        if node is None:
            continue

        print(
            f"[{node.node_id}] "
            f"Line {node.lineno} "
            f"{node.node_type}"
        )

        print(
            node.text
        )

        print()

    print("=" * 80)
    print("FORWARD ONLY")
    print("=" * 80)

    for node_id in sorted(
        forward_only
    ):

        node = node_map.get(
            node_id
        )

        if node is None:
            continue

        print(
            f"[{node.node_id}] "
            f"Line {node.lineno} "
            f"{node.node_type}"
        )

        print(
            node.text
        )

        print()

    print("=" * 80)
    print("BACKWARD ONLY")
    print("=" * 80)

    for node_id in sorted(
        backward_only
    ):

        node = node_map.get(
            node_id
        )

        if node is None:
            continue

        print(
            f"[{node.node_id}] "
            f"Line {node.lineno} "
            f"{node.node_type}"
        )

        print(
            node.text
        )

        print()

    print("=" * 80)
    print("OVERLAP")
    print("=" * 80)

    for node_id in sorted(
        overlap
    ):

        node = node_map.get(
            node_id
        )

        if node is None:
            continue

        print(
            f"[{node.node_id}] "
            f"Line {node.lineno} "
            f"{node.node_type}"
        )

        print(
            node.text
        )

        print()