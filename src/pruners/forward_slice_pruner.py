from collections import deque

from tesis.dataset.pruners.base_pruner import BasePruner
from tesis.dataset.pruners.utils import prune_cfg


class ForwardSlicePruner(BasePruner):

    def prune(
        self,
        sample
    ):

        cfg = sample.cfg
        seed_nodes = sample.seed_nodes

        #
        # Build forward graph.
        #
        graph = {}

        for src, dst in cfg["edges"]:

            graph.setdefault(
                src,
                []
            ).append(dst)

        #
        # Restrict traversal to the target function.
        #
        function_nodes = set(
            sample.function_nodes
        )

        #
        # Start from vulnerability seed nodes.
        #
        keep = set(seed_nodes)

        queue = deque(
            seed_nodes
        )

        #
        # Forward traversal.
        #
        while queue:

            node = queue.popleft()

            for child in graph.get(
                node,
                []
            ):

                #
                # Stay inside the function.
                #
                if child not in function_nodes:
                    continue

                if child in keep:
                    continue

                keep.add(
                    child
                )

                queue.append(
                    child
                )

        #
        # Build the pruned CFG.
        #
        sample.pruned_cfg = prune_cfg(
            cfg,
            keep
        )

        return sample