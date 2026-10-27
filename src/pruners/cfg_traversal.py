from collections import deque


VALID_DIRECTIONS = {"forward", "backward", "undirected"}


def traverse_cfg(
    cfg,
    seed_nodes,
    function_nodes,
    direction="forward",
    max_ast_hops=None,
):
    """
    Traverse a CFG within a target function's scope.

    Virtual nodes may be traversed as zero-cost intermediates but are
    excluded from the returned set.

    Args:
        cfg: Dictionary containing "nodes" and "edges".
             Nodes must expose node_id and lineno attributes.
        seed_nodes: Starting CFG node IDs.
        function_nodes: AST node IDs belonging to the target function.
        direction: "forward", "backward", or "undirected".
        max_ast_hops: Maximum number of AST-node hops, or None for
                      unlimited traversal. Virtual nodes cost zero hops.

    Returns:
        Set of reachable AST node IDs within function_nodes.

    Raises:
        ValueError: If direction or max_ast_hops is invalid.
    """
    _validate_direction(direction)
    _validate_hop_budget(max_ast_hops)

    node_by_id = _index_nodes(cfg)
    function_nodes = set(function_nodes or ())

    # Seeds must exist in the CFG and belong to the target function.
    seeds = set(seed_nodes or ()) & function_nodes & set(node_by_id)

    if not seeds:
        return set()

    adjacency = _build_adjacency(
        cfg["edges"],
        node_by_id,
        direction,
    )

    distances = {seed: 0 for seed in seeds}
    queue = deque(seeds)

    while queue:
        current = queue.popleft()
        current_distance = distances[current]

        for neighbor in adjacency[current]:
            next_distance = _next_distance(
                neighbor=neighbor,
                current_distance=current_distance,
                node_by_id=node_by_id,
                function_nodes=function_nodes,
            )

            # None means this node is outside the traversable scope.
            if next_distance is None:
                continue

            if (
                max_ast_hops is not None
                and next_distance > max_ast_hops
            ):
                continue

            previous_distance = distances.get(neighbor)

            if (
                previous_distance is not None
                and previous_distance <= next_distance
            ):
                continue

            distances[neighbor] = next_distance
            queue.append(neighbor)

    return set(distances) & function_nodes


def _validate_direction(direction):
    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            "Unsupported direction: {!r}. Expected one of: {}".format(
                direction,
                ", ".join(sorted(VALID_DIRECTIONS)),
            )
        )


def _validate_hop_budget(max_ast_hops):
    if max_ast_hops is None:
        return

    # bool is a subclass of int, so require an actual integer.
    if type(max_ast_hops) is not int or max_ast_hops < 0:
        raise ValueError(
            "max_ast_hops must be a non-negative integer or None"
        )


def _index_nodes(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("cfg must be a dictionary")

    nodes = cfg.get("nodes")
    edges = cfg.get("edges")

    if not isinstance(nodes, list):
        raise ValueError("cfg['nodes'] must be a list")

    if not isinstance(edges, list):
        raise ValueError("cfg['edges'] must be a list")

    node_by_id = {}

    for node in nodes:
        node_id = getattr(node, "node_id", None)
        lineno = getattr(node, "lineno", None)

        if node_id is None or not isinstance(lineno, int):
            continue

        node_by_id[node_id] = node

    return node_by_id


def _build_adjacency(edges, node_by_id, direction):
    adjacency = {
        node_id: set()
        for node_id in node_by_id
    }

    for edge in edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            continue

        src, dst = edge

        # Ignore edges that reference nonexistent nodes.
        if src not in node_by_id or dst not in node_by_id:
            continue

        if direction in {"forward", "undirected"}:
            adjacency[src].add(dst)

        if direction in {"backward", "undirected"}:
            adjacency[dst].add(src)

    return adjacency


def _next_distance(
    neighbor,
    current_distance,
    node_by_id,
    function_nodes,
):
    node = node_by_id[neighbor]

    # Virtual nodes are zero-cost intermediates.
    if node.lineno < 1:
        return current_distance

    # Ordinary AST nodes outside the target function are boundaries.
    if neighbor not in function_nodes:
        return None

    return current_distance + 1
