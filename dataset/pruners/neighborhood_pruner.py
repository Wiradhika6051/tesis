from collections import deque

from tesis.dataset.pruners.base_pruner import BasePruner
from tesis.dataset.pruners.utils import prune_cfg


class NeighborhoodPruner(BasePruner):

    def __init__(self, hops=2):
        self.hops = hops

    def prune(
        self,
        cfg,
        start_nodes
    ):

        #
        # No localization result.
        # Keep the entire graph.
        #
        if not start_nodes:

            return prune_cfg(
                cfg,
                {
                    node.node_id
                    for node in cfg["nodes"]
                }
            )

        keep = set(start_nodes)

        graph = {}

        for src, dst in cfg["edges"]:

            graph.setdefault(src, []).append(dst)
            graph.setdefault(dst, []).append(src)

        queue = deque(
            (node, 0)
            for node in start_nodes
        )

        while queue:

            node, depth = queue.popleft()

            if depth >= self.hops:
                continue

            for neighbor in graph.get(node, []):

                if neighbor in keep:
                    continue

                keep.add(neighbor)

                queue.append(
                    (
                        neighbor,
                        depth + 1
                    )
                )

        return prune_cfg(
            cfg,
            keep
        )