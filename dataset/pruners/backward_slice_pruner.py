from collections import deque

from tesis.dataset.pruners.base_pruner import BasePruner
from tesis.dataset.pruners.utils import prune_cfg


class BackwardSlicePruner(BasePruner):

    def prune(
        self,
        sample
    ):

        cfg = sample.cfg
        seed_nodes = sample.seed_nodes

        reverse_graph = {}

        for src, dst in cfg["edges"]:

            reverse_graph.setdefault(
                dst,
                []
            ).append(src)

        keep = set(seed_nodes)

        queue = deque(seed_nodes)

        while queue:

            node = queue.popleft()

            for parent in reverse_graph.get(node, []):

                if parent in keep:
                    continue

                keep.add(parent)

                queue.append(parent)

        sample.pruned_cfg = prune_cfg(
            cfg,
            keep
        )

        return sample