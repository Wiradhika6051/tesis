from collections import deque

from src.pruners.base_pruner import BasePruner
from src.pruners.utils import prune_cfg


class NeighborhoodPruner(BasePruner):

    def __init__(
        self,
        hops=2
    ):

        self.hops = hops

    def prune(
        self,
        sample
    ):

        cfg = sample.cfg
        seed_nodes = sample.seed_nodes

        #
        # Restrict traversal to the target function.
        #
        function_nodes = set(
            sample.function_nodes
        )

        #
        # No seeds.
        #
        if not seed_nodes:

            keep = function_nodes

            sample.pruned_cfg = prune_cfg(
                cfg,
                keep
            )

            return sample

        #
        # Build undirected graph.
        #
        graph = {}

        for src, dst in cfg["edges"]:

            if (
                src not in function_nodes
                or
                dst not in function_nodes
            ):
                continue

            graph.setdefault(
                src,
                []
            ).append(dst)

            graph.setdefault(
                dst,
                []
            ).append(src)

        #
        # Only use seeds that belong
        # to the target function.
        #
        seed_nodes = [
            node
            for node in seed_nodes
            if node in function_nodes
        ]

        keep = set(
            seed_nodes
        )

        queue = deque(
            (
                node,
                0
            )
            for node in seed_nodes
        )

        #
        # BFS neighborhood.
        #
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

        sample.pruned_cfg = prune_cfg(
            cfg,
            keep
        )

        return sample