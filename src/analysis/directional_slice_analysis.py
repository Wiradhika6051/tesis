from collections import Counter, defaultdict


def get_node_map(cfg):
    """
    Convert CFG nodes into:

        {
            node_id: node
        }
    """

    if cfg is None:
        return {}

    nodes = cfg.get(
        "nodes",
        []
    )

    return {
        node.node_id: node
        for node in nodes
    }


def get_node_type(node):

    if node is None:
        return "UNKNOWN"

    return getattr(
        node,
        "node_type",
        "UNKNOWN"
    )


def get_node_tokens(node):

    if node is None:
        return []

    text = getattr(
        node,
        "text",
        ""
    )

    if text is None:
        return []

    return text.split()


def get_slice_node_ids(slice_result):
    """
    Accept either:

    set(...)
    list(...)
    or a dictionary containing nodes.
    """

    if slice_result is None:
        return set()

    if isinstance(
        slice_result,
        dict
    ):

        nodes = slice_result.get(
            "nodes",
            []
        )

    else:

        nodes = slice_result

    result = set()

    for node in nodes:

        if isinstance(
            node,
            int
        ):

            result.add(
                node
            )

        else:

            node_id = getattr(
                node,
                "node_id",
                None
            )

            if node_id is not None:

                result.add(
                    node_id
                )

    return result


def analyze_directional_slice_difference(
    samples,
    top_samples=20
):

    overall = {

        "samples": 0,

        "forward_nodes": 0,
        "backward_nodes": 0,

        "overlap_nodes": 0,

        "forward_only_nodes": 0,
        "backward_only_nodes": 0,

        "forward_only_types": Counter(),
        "backward_only_types": Counter(),
        "overlap_types": Counter(),

        "forward_only_tokens": 0,
        "backward_only_tokens": 0,

    }

    by_label = defaultdict(

        lambda: {

            "samples": 0,

            "forward_nodes": 0,
            "backward_nodes": 0,

            "overlap_nodes": 0,

            "forward_only_nodes": 0,
            "backward_only_nodes": 0,

            "forward_only_types": Counter(),
            "backward_only_types": Counter(),
            "overlap_types": Counter(),

            "forward_only_tokens": 0,
            "backward_only_tokens": 0,

        }

    )

    sample_results = []

    for sample in samples:

        #
        # Retrieve slice analysis.
        #
        analysis = getattr(
            sample,
            "slice_analysis",
            None
        )

        if analysis is None:
            continue

        forward_ids = set(
            analysis["forward_node_ids"]
        )

        backward_ids = set(
            analysis["backward_node_ids"]
        )
        #
        # CFG node lookup.
        #
        node_map = get_node_map(
            sample.cfg
        )

        #
        # Compare.
        #
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

        label = sample.label

        overall["samples"] += 1

        overall["forward_nodes"] += (
            len(forward_ids)
        )

        overall["backward_nodes"] += (
            len(backward_ids)
        )

        overall["overlap_nodes"] += (
            len(overlap)
        )

        overall["forward_only_nodes"] += (
            len(forward_only)
        )

        overall["backward_only_nodes"] += (
            len(backward_only)
        )

        by_label[label]["samples"] += 1

        by_label[label]["forward_nodes"] += (
            len(forward_ids)
        )

        by_label[label]["backward_nodes"] += (
            len(backward_ids)
        )

        by_label[label]["overlap_nodes"] += (
            len(overlap)
        )

        by_label[label]["forward_only_nodes"] += (
            len(forward_only)
        )

        by_label[label]["backward_only_nodes"] += (
            len(backward_only)
        )

        #
        # Analyze forward-only nodes.
        #
        for node_id in forward_only:

            node = node_map.get(
                node_id
            )

            node_type = get_node_type(
                node
            )

            tokens = get_node_tokens(
                node
            )

            overall[
                "forward_only_types"
            ][node_type] += 1

            overall[
                "forward_only_tokens"
            ] += len(tokens)

            by_label[label][
                "forward_only_types"
            ][node_type] += 1

            by_label[label][
                "forward_only_tokens"
            ] += len(tokens)

        #
        # Analyze backward-only nodes.
        #
        for node_id in backward_only:

            node = node_map.get(
                node_id
            )

            node_type = get_node_type(
                node
            )

            tokens = get_node_tokens(
                node
            )

            overall[
                "backward_only_types"
            ][node_type] += 1

            overall[
                "backward_only_tokens"
            ] += len(tokens)

            by_label[label][
                "backward_only_types"
            ][node_type] += 1

            by_label[label][
                "backward_only_tokens"
            ] += len(tokens)

        #
        # Analyze overlap.
        #
        for node_id in overlap:

            node = node_map.get(
                node_id
            )

            node_type = get_node_type(
                node
            )

            overall[
                "overlap_types"
            ][node_type] += 1

            by_label[label][
                "overlap_types"
            ][node_type] += 1

        #
        # Store sample-level result.
        #
        difference = abs(
            len(forward_only)
            -
            len(backward_only)
        )

        sample_results.append({

            "sample": sample,

            "label": label,

            "forward": len(
                forward_ids
            ),

            "backward": len(
                backward_ids
            ),

            "overlap": len(
                overlap
            ),

            "forward_only": len(
                forward_only
            ),

            "backward_only": len(
                backward_only
            ),

            "difference": difference,

        })

    #
    # Print results.
    #
    print(
        "=" * 80
    )

    print(
        "DIRECTIONAL SLICE DIFFERENCE ANALYSIS"
    )

    print(
        "=" * 80
    )

    print()

    samples_count = overall[
        "samples"
    ]

    if samples_count == 0:

        print(
            "No valid slice analysis found."
        )

        return overall

    print(
        f"Samples analyzed : {samples_count}"
    )

    print()

    print(
        "OVERALL"
    )

    print(
        "-" * 60
    )

    print(
        f"Average Forward Nodes  : "
        f"{overall['forward_nodes'] / samples_count:.2f}"
    )

    print(
        f"Average Backward Nodes : "
        f"{overall['backward_nodes'] / samples_count:.2f}"
    )

    print(
        f"Average Overlap Nodes  : "
        f"{overall['overlap_nodes'] / samples_count:.2f}"
    )

    print(
        f"Average Forward-Only   : "
        f"{overall['forward_only_nodes'] / samples_count:.2f}"
    )

    print(
        f"Average Backward-Only  : "
        f"{overall['backward_only_nodes'] / samples_count:.2f}"
    )

    print()

    print(
        "FORWARD-ONLY NODE TYPES"
    )

    print(
        "-" * 60
    )

    for node_type, count in (
        overall[
            "forward_only_types"
        ].most_common()
    ):

        print(
            f"{node_type:<25} "
            f"{count}"
        )

    print()

    print(
        "BACKWARD-ONLY NODE TYPES"
    )

    print(
        "-" * 60
    )

    for node_type, count in (
        overall[
            "backward_only_types"
        ].most_common()
    ):

        print(
            f"{node_type:<25} "
            f"{count}"
        )

    print()

    print(
        "OVERLAP NODE TYPES"
    )

    print(
        "-" * 60
    )

    for node_type, count in (
        overall[
            "overlap_types"
        ].most_common()
    ):

        print(
            f"{node_type:<25} "
            f"{count}"
        )

    print()

    print(
        "=" * 80
    )

    print(
        "BY LABEL"
    )

    print(
        "=" * 80
    )

    for label in sorted(
        by_label.keys()
    ):

        stats = by_label[
            label
        ]

        count = stats[
            "samples"
        ]

        print()

        print(
            f"Label {label}"
        )

        print(
            "-" * 60
        )

        print(
            f"Samples             : {count}"
        )

        print(
            f"Avg Forward Nodes   : "
            f"{stats['forward_nodes'] / count:.2f}"
        )

        print(
            f"Avg Backward Nodes  : "
            f"{stats['backward_nodes'] / count:.2f}"
        )

        print(
            f"Avg Overlap         : "
            f"{stats['overlap_nodes'] / count:.2f}"
        )

        print(
            f"Avg Forward-Only    : "
            f"{stats['forward_only_nodes'] / count:.2f}"
        )

        print(
            f"Avg Backward-Only   : "
            f"{stats['backward_only_nodes'] / count:.2f}"
        )

        print()

        print(
            "Top Forward-Only Types"
        )

        for node_type, value in (
            stats[
                "forward_only_types"
            ].most_common(10)
        ):

            print(
                f"  {node_type:<25} "
                f"{value}"
            )

        print()

        print(
            "Top Backward-Only Types"
        )

        for node_type, value in (
            stats[
                "backward_only_types"
            ].most_common(10)
        ):

            print(
                f"  {node_type:<25} "
                f"{value}"
            )

    #
    # Extreme samples.
    #
    sample_results.sort(

        key=lambda x:
            x["difference"],

        reverse=True

    )

    print()

    print(
        "=" * 80
    )

    print(
        "LARGEST DIRECTIONAL DIFFERENCES"
    )

    print(
        "=" * 80
    )

    for result in sample_results[
        :top_samples
    ]:

        sample = result[
            "sample"
        ]

        print()

        print(
            f"Label: {result['label']}"
        )

        print(
            f"Repo: {sample.repo}"
        )

        print(
            f"File: {sample.file_path}"
        )

        print(
            f"Forward Nodes: "
            f"{result['forward']}"
        )

        print(
            f"Backward Nodes: "
            f"{result['backward']}"
        )

        print(
            f"Overlap: "
            f"{result['overlap']}"
        )

        print(
            f"Forward Only: "
            f"{result['forward_only']}"
        )

        print(
            f"Backward Only: "
            f"{result['backward_only']}"
        )

        print(
            f"Difference: "
            f"{result['difference']}"
        )

    return {

        "overall": overall,

        "by_label": dict(
            by_label
        ),

        "samples": sample_results

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

        "forward_only":
            len(forward_only),

        "backward_only":
            len(backward_only),

        "overlap":
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
