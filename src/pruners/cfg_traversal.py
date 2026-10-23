
from collections import deque


def traverse_cfg(
    cfg,
    seed_nodes,
    function_nodes,
    direction="forward",
    max_ast_hops=None,
):
    """
    Traverse a CFG while allowing virtual nodes as intermediates.

    Args:
        cfg: {"nodes": [...], "edges": [...]}
        seed_nodes: Starting CFG node IDs.
        function_nodes: AST node IDs in the target function.
        direction: "forward", "backward", or "undirected".
        max_ast_hops: Optional distance measured in AST-node hops.
                      Virtual nodes do not consume hops.

    Returns:
        Set of reachable AST node IDs in function_nodes.
    """
    if direction not in {"forward", "backward", "undirected"}:
        raise ValueError(f"Unsupported direction: {direction}")

    node_by_id = {
        node.node_id: node
        for node in cfg["nodes"]
    }

    function_nodes = set(function_nodes)
    seeds = set(seed_nodes) & function_nodes

    if not seeds:
        return set()

    adjacency = {node_id: set() for node_id in node_by_id}

    for src, dst in cfg["edges"]:
        if src not in node_by_id or dst not in node_by_id:
            continue

        if direction in {"forward", "undirected"}:
            adjacency[src].add(dst)

        if direction in {"backward", "undirected"}:
            adjacency[dst].add(src)

    def is_virtual(node_id):
        node = node_by_id[node_id]
        return node.lineno < 1

    # State tracks the minimum AST-hop distance to each node.
    distances = {seed: 0 for seed in seeds}
    queue = deque(seeds)

    while queue:
        current = queue.popleft()
        current_distance = distances[current]

        for neighbor in adjacency[current]:
            # A virtual node may be traversed but is never retained
            # as an AST node.
            if is_virtual(neighbor):
                next_distance = current_distance

            # An AST node outside the target function is a boundary.
            elif neighbor in function_nodes:
                next_distance = current_distance + 1

            else:
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

    # Return only AST nodes eligible for the target function.
    return set(distances) & function_nodes
