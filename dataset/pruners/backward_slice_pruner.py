from collections import deque

from tesis.dataset.pruners.base_pruner import BasePruner
from tesis.dataset.pruners.utils import prune_cfg

class BackwardSlicePruner(
    BasePruner
):

    def prune(
        self,
        cfg,
        snippet
    ):

        targets = cfg[
            "target_nodes"
        ]

        reverse_graph = {}

        for src, dst in cfg["edges"]:

            reverse_graph.setdefault(
                dst,
                []
            ).append(
                src
            )

        keep = set(
            targets
        )

        queue = deque(
            targets
        )

        while queue:

            node = queue.popleft()

            for parent in reverse_graph.get(
                node,
                []
            ):

                if parent in keep:
                    continue

                keep.add(
                    parent
                )

                queue.append(
                    parent
                )

        return prune_cfg(
            cfg,
            keep
        )