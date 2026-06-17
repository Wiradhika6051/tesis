from collections import deque

from tesis.dataset.pruners.base_pruner import BasePruner
from tesis.dataset.pruners.utils import prune_cfg

class NeighborhoodPruner(
    BasePruner
):

    def __init__(
        self,
        hops=2
    ):

        self.hops = hops

    def prune(
        self,
        cfg,
        snippet
    ):

        targets = cfg[
            "target_nodes"
        ]

        keep = set(
            targets
        )

        graph = {}

        for src, dst in cfg["edges"]:

            graph.setdefault(
                src,
                []
            ).append(
                dst
            )

            graph.setdefault(
                dst,
                []
            ).append(
                src
            )

        queue = deque()

        for node in targets:

            queue.append(
                (
                    node,
                    0
                )
            )

        while queue:

            node, depth = queue.popleft()

            if depth >= self.hops:
                continue

            for neighbor in graph.get(
                node,
                []
            ):

                if neighbor in keep:
                    continue

                keep.add(
                    neighbor
                )

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