from collections import Counter
import statistics


def compare_slices(
    samples
):
    """
    Compare backward and forward slices
    for the same samples.

    This analysis does NOT modify samples.
    """

    results = []

    for sample in samples:

        cfg = sample.cfg

        function_nodes = set(
            sample.function_nodes
        )

        seed_nodes = set(
            sample.seed_nodes
        )

        #
        # Build backward slice.
        #
        reverse_graph = {}

        for src, dst in cfg["edges"]:

            reverse_graph.setdefault(
                dst,
                []
            ).append(src)

        backward = set(
            seed_nodes
        )

        queue = list(
            seed_nodes
        )

        while queue:

            node = queue.pop()

            for parent in reverse_graph.get(
                node,
                []
            ):

                #
                # Stay inside localized
                # function.
                #
                if parent not in function_nodes:
                    continue

                if parent in backward:
                    continue

                backward.add(
                    parent
                )

                queue.append(
                    parent
                )

        #
        # Build forward slice.
        #
        graph = {}

        for src, dst in cfg["edges"]:

            graph.setdefault(
                src,
                []
            ).append(dst)

        forward = set(
            seed_nodes
        )

        queue = list(
            seed_nodes
        )

        while queue:

            node = queue.pop()

            for child in graph.get(
                node,
                []
            ):

                #
                # Stay inside localized
                # function.
                #
                if child not in function_nodes:
                    continue

                if child in forward:
                    continue

                forward.add(
                    child
                )

                queue.append(
                    child
                )

        #
        # Compare.
        #
        common = (
            backward &
            forward
        )

        backward_only = (
            backward -
            forward
        )

        forward_only = (
            forward -
            backward
        )

        results.append({

            "sample": sample,

            "label": sample.label,

            "function_nodes":
                function_nodes,

            "seed_nodes":
                seed_nodes,

            "backward":
                backward,

            "forward":
                forward,

            "common":
                common,

            "backward_only":
                backward_only,

            "forward_only":
                forward_only,

        })

    return results


def print_slice_comparison_summary(
    results
):
    """
    Print aggregate statistics comparing
    backward and forward slices.
    """

    if not results:

        print(
            "No slice comparison results."
        )

        return

    print(
        "=" * 70
    )

    print(
        "BACKWARD vs FORWARD SLICE ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"Samples analyzed : "
        f"{len(results)}"
    )

    #
    # --------------------------------------------------
    # Overall
    # --------------------------------------------------
    #

    backward_sizes = [
        len(r["backward"])
        for r in results
    ]

    forward_sizes = [
        len(r["forward"])
        for r in results
    ]

    common_sizes = [
        len(r["common"])
        for r in results
    ]

    backward_only_sizes = [
        len(r["backward_only"])
        for r in results
    ]

    forward_only_sizes = [
        len(r["forward_only"])
        for r in results
    ]

    print(
        "\n## Overall"
    )

    print(
        f"Average Backward Nodes : "
        f"{statistics.mean(backward_sizes):.2f}"
    )

    print(
        f"Average Forward Nodes  : "
        f"{statistics.mean(forward_sizes):.2f}"
    )

    print(
        f"Average Common Nodes   : "
        f"{statistics.mean(common_sizes):.2f}"
    )

    print(
        f"Average Backward Only  : "
        f"{statistics.mean(backward_only_sizes):.2f}"
    )

    print(
        f"Average Forward Only   : "
        f"{statistics.mean(forward_only_sizes):.2f}"
    )

    #
    # Similarity.
    #
    jaccard = []

    for r in results:

        union = (
            r["backward"] |
            r["forward"]
        )

        if union:

            jaccard.append(
                len(r["common"]) /
                len(union)
            )

    if jaccard:

        print(
            f"Average Slice Jaccard : "
            f"{statistics.mean(jaccard):.2%}"
        )

    #
    # --------------------------------------------------
    # By label
    # --------------------------------------------------
    #

    labels = sorted(
        set(
            r["label"]
            for r in results
        )
    )

    print(
        "\n## By Label"
    )

    for label in labels:

        group = [
            r
            for r in results
            if r["label"] == label
        ]

        print(
            f"\nLabel {label}"
        )

        print(
            f"Samples                : "
            f"{len(group)}"
        )

        print(
            f"Average Backward       : "
            f"{statistics.mean(len(r['backward'])for r in group): .2f}"
        )

        print(
            f"Average Forward        : "
            f"{statistics.mean(len(r['forward'])for r in group): .2f}"
        )

        print(
            f"Average Common         : "
            f"{statistics.mean(len(r['common']) for r in group): .2f}"
        )

        print(
            f"Average Backward Only  : "
            f"{statistics.mean(len(r['backward_only']) for r in group): .2f}"
        )

        print(
            f"Average Forward Only   : "
            f"{statistics.mean(len(r['forward_only']) for r in group): .2f}"
        )

    #
    # --------------------------------------------------
    # Node type analysis
    # --------------------------------------------------
    #

    print(
        "\n## Node Type Distribution"
    )

    backward_types = Counter()
    forward_types = Counter()
    common_types = Counter()

    for r in results:

        node_map = {
            node.node_id: node
            for node in r["sample"].cfg["nodes"]
        }

        for node_id in r["backward_only"]:

            node = node_map.get(
                node_id
            )

            if node is not None:

                backward_types[
                    node.node_type
                ] += 1

        for node_id in r["forward_only"]:

            node = node_map.get(
                node_id
            )

            if node is not None:

                forward_types[
                    node.node_type
                ] += 1

        for node_id in r["common"]:

            node = node_map.get(
                node_id
            )

            if node is not None:

                common_types[
                    node.node_type
                ] += 1

    print(
        "\nBackward-only nodes:"
    )

    for node_type, count in (
        backward_types.most_common()
    ):

        print(
            f"{node_type:<20} "
            f"{count}"
        )

    print(
        "\nForward-only nodes:"
    )

    for node_type, count in (
        forward_types.most_common()
    ):

        print(
            f"{node_type:<20} "
            f"{count}"
        )

    print(
        "\nCommon nodes:"
    )

    for node_type, count in (
        common_types.most_common()
    ):

        print(
            f"{node_type:<20} "
            f"{count}"
        )


def print_worst_slice_differences(
    results,
    limit=20
):
    """
    Show samples where backward and forward
    slices differ the most.
    """

    ranked = sorted(
        results,
        key=lambda r: (
            len(r["backward_only"]) +
            len(r["forward_only"])
        ),
        reverse=True
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "LARGEST BACKWARD / FORWARD DIFFERENCES"
    )

    print(
        "=" * 70
    )

    for r in ranked[:limit]:

        sample = r["sample"]

        difference = (
            len(r["backward_only"]) +
            len(r["forward_only"])
        )

        print(
            f"\nRepo : {sample.repo}"
        )

        print(
            f"File : {sample.file_path}"
        )

        print(
            f"Label: {sample.label}"
        )

        print(
            f"Seeds           : "
            f"{len(r['seed_nodes'])}"
        )

        print(
            f"Backward        : "
            f"{len(r['backward'])}"
        )

        print(
            f"Forward         : "
            f"{len(r['forward'])}"
        )

        print(
            f"Common          : "
            f"{len(r['common'])}"
        )

        print(
            f"Backward only   : "
            f"{len(r['backward_only'])}"
        )

        print(
            f"Forward only    : "
            f"{len(r['forward_only'])}"
        )

        print(
            f"Total difference: "
            f"{difference}"
        )

def print_slice_difference_details(
    results,
    limit=10
):
    """
    Print detailed CFG information for samples
    where backward and forward slices differ most.

    For each selected sample, show:

        - Seed nodes
        - Common nodes
        - Backward-only nodes
        - Forward-only nodes

    Each node includes:

        node ID
        line number
        node type
        source code
    """

    if not results:

        print(
            "No slice comparison results."
        )

        return

    #
    # Rank samples by total difference.
    #
    ranked = sorted(

        results,

        key=lambda r: (
            len(r["backward_only"]) +
            len(r["forward_only"])
        ),

        reverse=True
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "DETAILED BACKWARD / FORWARD SLICE DIFFERENCES"
    )

    print(
        "=" * 80
    )

    for index, result in enumerate(
        ranked[:limit],
        start=1
    ):

        sample = result["sample"]

        node_map = {

            node.node_id: node

            for node in sample.cfg["nodes"]

        }

        print(
            "\n" + "-" * 80
        )

        print(
            f"Sample #{index}"
        )

        print(
            "-" * 80
        )

        print(
            f"Repo       : {sample.repo}"
        )

        print(
            f"File       : {sample.file_path}"
        )

        print(
            f"Label      : {sample.label}"
        )

        print(
            f"Seeds      : "
            f"{len(result['seed_nodes'])}"
        )

        print(
            f"Backward   : "
            f"{len(result['backward'])}"
        )

        print(
            f"Forward    : "
            f"{len(result['forward'])}"
        )

        print(
            f"Common     : "
            f"{len(result['common'])}"
        )

        print(
            f"Backward-only : "
            f"{len(result['backward_only'])}"
        )

        print(
            f"Forward-only  : "
            f"{len(result['forward_only'])}"
        )

        #
        # --------------------------------------------------
        # Seeds
        # --------------------------------------------------
        #

        _print_nodes(
            title="SEED NODES",
            node_ids=result["seed_nodes"],
            node_map=node_map
        )

        #
        # --------------------------------------------------
        # Common
        # --------------------------------------------------
        #

        _print_nodes(
            title="COMMON NODES",
            node_ids=result["common"],
            node_map=node_map
        )

        #
        # --------------------------------------------------
        # Backward-only
        # --------------------------------------------------
        #

        _print_nodes(
            title="BACKWARD-ONLY NODES",
            node_ids=result["backward_only"],
            node_map=node_map
        )

        #
        # --------------------------------------------------
        # Forward-only
        # --------------------------------------------------
        #

        _print_nodes(
            title="FORWARD-ONLY NODES",
            node_ids=result["forward_only"],
            node_map=node_map
        )


def _print_nodes(
    title,
    node_ids,
    node_map
):
    """
    Print node information for a set of CFG nodes.
    """

    print(
        f"\n### {title}"
    )

    if not node_ids:

        print(
            "None"
        )

        return

    #
    # Sort primarily by source line.
    #
    nodes = []

    for node_id in node_ids:

        node = node_map.get(
            node_id
        )

        if node is not None:

            nodes.append(
                node
            )

    nodes.sort(
        key=lambda node: (
            getattr(
                node,
                "lineno",
                -1
            ),
            node.node_id
        )
    )

    for node in nodes:

        line = getattr(
            node,
            "lineno",
            -1
        )

        node_type = getattr(
            node,
            "node_type",
            "UNKNOWN"
        )

        text = getattr(
            node,
            "text",
            ""
        )

        #
        # Prevent huge nodes from flooding
        # the notebook output.
        #
        text = text.strip()

        if len(text) > 300:

            text = (
                text[:300]
                + "..."
            )

        #
        # Keep multiline source readable.
        #
        text = text.replace(
            "\n",
            "\\n"
        )

        print(
            f"Node {node.node_id:>4} | "
            f"Line {line:>4} | "
            f"{node_type:<20} | "
            f"{text}"
        )