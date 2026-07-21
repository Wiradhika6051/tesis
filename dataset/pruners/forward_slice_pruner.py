from collections import deque

from tesis.dataset.pruners.base_pruner import BasePruner
from tesis.dataset.pruners.utils import prune_cfg


class ForwardSlicePruner(BasePruner):

    def prune(
        self,
        cfg,
        seed_nodes
    ):

        graph = {}

        for src, dst in cfg["edges"]:

            graph.setdefault(
                src,
                []
            ).append(dst)

        keep = set(seed_nodes)

        queue = deque(seed_nodes)

        while queue:

            node = queue.popleft()

            for child in graph.get(node, []):

                if child in keep:
                    continue

                keep.add(child)

                queue.append(child)

        return prune_cfg(
            cfg,
            keep
        )