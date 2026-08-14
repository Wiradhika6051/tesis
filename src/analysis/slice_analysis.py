from collections import deque


def get_slice_nodes(
    cfg,
    seed_nodes,
    function_nodes,
    forward
):
    """
    Calculate the nodes retained by a
    forward or backward slice.

    This does NOT modify the sample.
    """

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


def analyze_slice(
    sample
):
    """
    Analyze forward and backward slices
    before pruning.
    """

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

    function_size = len(
        function_nodes
    )

    return {
        "repo":
            sample.repo,

        "file":
            sample.file_path,

        "label":
            sample.label,

        "seed_nodes":
            len(seed_nodes),

        "function_nodes":
            function_size,

        "backward_nodes":
            len(backward_nodes),

        "forward_nodes":
            len(forward_nodes),

        "backward_ratio":
            len(backward_nodes)
            /
            function_size,

        "forward_ratio":
            len(forward_nodes)
            /
            function_size
    }