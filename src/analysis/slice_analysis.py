from collections import deque


def get_slice_nodes(
    cfg,
    seed_nodes,
    function_nodes,
    forward
):

    allowed = set(
        function_nodes
    )

    #
    # Only seeds belonging to the
    # localized function are valid.
    #
    valid_seeds = (
        set(seed_nodes)
        &
        allowed
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

    #
    # Start only from valid seeds.
    #
    keep = set(
        valid_seeds
    )

    queue = deque(
        valid_seeds
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

def analyze_slice(
    sample
):

    cfg = sample.cfg

    seed_nodes = set(
        sample.seed_nodes
    )

    function_nodes = set(
        sample.function_nodes
    )

    if not seed_nodes:
        return None

    if not function_nodes:
        return None

    #
    # IMPORTANT DIAGNOSTIC
    #
    seeds_outside_function = (
        seed_nodes
        -
        function_nodes
    )

    #
    # Calculate slices
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
    # Make sure they are sets.
    #
    backward_nodes = set(
        backward_nodes
    )

    forward_nodes = set(
        forward_nodes
    )

    function_size = len(
        function_nodes
    )

    result = {

        "repo":
            sample.repo,

        "file":
            sample.file_path,

        "label":
            sample.label,

        #
        # Counts
        #
        "seed_nodes":
            len(seed_nodes),

        "seeds_outside_function":
            len(seeds_outside_function),

        "function_nodes":
            function_size,

        "backward_nodes":
            len(backward_nodes),

        "forward_nodes":
            len(forward_nodes),

        #
        # Ratios
        #
        "backward_ratio":
            len(backward_nodes)
            /
            function_size,

        "forward_ratio":
            len(forward_nodes)
            /
            function_size,

        #
        # Actual node IDs.
        #
        "seed_node_ids":
            sorted(seed_nodes),

        "function_node_ids":
            sorted(function_nodes),

        "backward_node_ids":
            sorted(backward_nodes),

        "forward_node_ids":
            sorted(forward_nodes)
    }

    #
    # Print suspicious samples.
    #
    if (
        len(backward_nodes)
        >
        function_size
    ):

        print()
        print("=" * 60)
        print("SUSPICIOUS SLICE")
        print("=" * 60)

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

        print(
            "Seed nodes:",
            seed_nodes
        )

        print(
            "Function nodes:",
            function_nodes
        )

        print(
            "Seeds outside function:",
            seeds_outside_function
        )

        print(
            "Function size:",
            function_size
        )

        print(
            "Backward size:",
            len(backward_nodes)
        )

        print(
            "Forward size:",
            len(forward_nodes)
        )

    return result