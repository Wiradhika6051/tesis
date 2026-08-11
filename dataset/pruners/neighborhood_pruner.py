from collections import deque

from tesis.dataset.pruners.base_pruner import BasePruner
from tesis.dataset.pruners.utils import prune_cfg


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
        # Restrict traversal to the localized function.
        #
        function_nodes = set(
            sample.function_nodes
        )

        #
        # No localization result.
        #
        # Keep the entire localized function.
        #
        if not seed_nodes:

            keep = function_nodes

            sample.pruned_cfg = prune_cfg(
                cfg,
                keep
            )

            return sample

        #
        # Build an undirected CFG graph.
        #
        graph = {}

        for src, dst in cfg["edges"]:

            graph.setdefault(
                src,
                []
            ).append(dst)

            graph.setdefault(
                dst,
                []
            ).append(src)

        #
        # Start from vulnerability seed nodes.
        #
        keep = set()

        queue = deque()

        for seed in seed_nodes:

            #
            # Ignore seeds outside the
            # localized function.
            #
            if seed not in function_nodes:
                continue

            keep.add(
                seed
            )

            queue.append(
                (
                    seed,
                    0
                )
            )

        #
        # Neighborhood traversal.
        #
        while queue:

            node, depth = queue.popleft()

            #
            # Stop expanding once the
            # requested number of hops
            # has been reached.
            #
            if depth >= self.hops:
                continue

            for neighbor in graph.get(
                node,
                []
            ):

                #
                # Stay inside the function.
                #
                if neighbor not in function_nodes:
                    continue

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

        #
        # Build the pruned CFG.
        #
        sample.pruned_cfg = prune_cfg(
            cfg,
            keep
        )

        return sample