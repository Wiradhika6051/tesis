from statistics import mean, median

from src.type.Sample import Sample

from collections import deque

def get_slice_nodes(
    cfg,
    seed_nodes,
    function_nodes,
    forward=True
):

    allowed = set(
        function_nodes
    )

    graph = {}

    for src, dst in cfg["edges"]:

        if forward:

            graph.setdefault(
                src,
                []
            ).append(dst)

        else:

            graph.setdefault(
                dst,
                []
            ).append(src)

    keep = set(
        seed_nodes
    )

    queue = deque(
        seed_nodes
    )

    while queue:

        node = queue.popleft()

        for neighbor in graph.get(
            node,
            []
        ):

            if neighbor not in allowed:
                continue

            if neighbor in keep:
                continue

            keep.add(
                neighbor
            )

            queue.append(
                neighbor
            )

    return keep

def get_forward_nodes(
    cfg,
    seed_nodes
):

    graph = {}

    for src, dst in cfg["edges"]:

        graph.setdefault(
            src,
            []
        ).append(dst)

    keep = set(seed_nodes)

    queue = deque(seed_nodes)

    while queue:

        node = queue.popleft()

        for child in graph.get(
            node,
            []
        ):

            if child in keep:
                continue

            keep.add(child)

            queue.append(child)

    return keep

def get_backward_nodes(
    cfg,
    seed_nodes
):

    reverse_graph = {}

    for src, dst in cfg["edges"]:

        reverse_graph.setdefault(
            dst,
            []
        ).append(src)

    keep = set(seed_nodes)

    queue = deque(seed_nodes)

    while queue:

        node = queue.popleft()

        for parent in reverse_graph.get(
            node,
            []
        ):

            if parent in keep:
                continue

            keep.add(parent)

            queue.append(parent)

    return keep


def analyze_pruning(
    original_samples: list[Sample],
    pruned_samples: list[Sample],
    name: str
):
    """
    Compare original CFGs against their pruned CFGs.

    original_samples:
        Samples before pruning.

    pruned_samples:
        Corresponding samples after pruning.
    """

    results = []

    for original, pruned in zip(
        original_samples,
        pruned_samples
    ):

        original_nodes = original.cfg["nodes"]
        pruned_nodes = pruned.pruned_cfg["nodes"]

        #
        # Node counts
        #
        original_node_count = len(
            original_nodes
        )

        pruned_node_count = len(
            pruned_nodes
        )

        #
        # Token counts
        #
        original_token_count = sum(
            len(node.tokens)
            for node in original_nodes
            if hasattr(node, "tokens")
        )

        pruned_token_count = sum(
            len(node.tokens)
            for node in pruned_nodes
            if hasattr(node, "tokens")
        )

        #
        # Retention ratios
        #
        if original_node_count > 0:

            node_retention = (
                pruned_node_count
                /
                original_node_count
            )

        else:

            node_retention = 0.0

        if original_token_count > 0:

            token_retention = (
                pruned_token_count
                /
                original_token_count
            )

        else:

            token_retention = 0.0

        #
        # Seed information
        #
        seed_nodes = set(
            original.seed_nodes
        )

        pruned_node_ids = {
            node.node_id
            for node in pruned_nodes
        }

        surviving_seeds = (
            seed_nodes
            &
            pruned_node_ids
        )

        #
        # Function information
        #
        function_nodes = set(
            original.function_nodes
        )

        function_node_count = len(
            function_nodes
        )

        results.append(
            {
                "label":
                    original.label,

                "original_nodes":
                    original_node_count,

                "pruned_nodes":
                    pruned_node_count,

                "node_retention":
                    node_retention,

                "original_tokens":
                    original_token_count,

                "pruned_tokens":
                    pruned_token_count,

                "token_retention":
                    token_retention,

                "seed_count":
                    len(seed_nodes),

                "surviving_seed_count":
                    len(surviving_seeds),

                "function_nodes":
                    function_node_count
            }
        )

    #
    # Summary
    #
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Samples              : {len(results)}"
    )

    if not results:
        return results

    print(
        f"Original nodes       : "
        f"{mean(r['original_nodes'] for r in results):.2f}"
    )

    print(
        f"Pruned nodes         : "
        f"{mean(r['pruned_nodes'] for r in results):.2f}"
    )

    print(
        f"Node retention       : "
        f"{mean(r['node_retention'] for r in results) * 100:.2f}%"
    )

    print(
        f"Original tokens      : "
        f"{mean(r['original_tokens'] for r in results):.2f}"
    )

    print(
        f"Pruned tokens        : "
        f"{mean(r['pruned_tokens'] for r in results):.2f}"
    )

    print(
        f"Token retention      : "
        f"{mean(r['token_retention'] for r in results) * 100:.2f}%"
    )

    print(
        f"Avg seed nodes       : "
        f"{mean(r['seed_count'] for r in results):.2f}"
    )

    print(
        f"Seed survival        : "
        f"{mean(r['surviving_seed_count'] == r['seed_count'] for r in results) * 100:.2f}%"
    )

    return results