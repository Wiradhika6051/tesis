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

        graph = {}

        for src, dst in cfg["edges"]:

            graph.setdefault(
                src,
                []
            ).append(dst)

        valid_seeds = (
            set(seed_nodes)
            &
            set(function_nodes)
        )
        
        keep = set(
            valid_seeds
        )
        
        queue = deque(
            valid_seeds
        )

        while queue:

            node = queue.popleft()

            for child in graph.get(node, []):

                if child in keep:
                    continue

                keep.add(child)

                queue.append(child)

        sample.pruned_cfg = prune_cfg(
            cfg,
            keep
        )

        return sample