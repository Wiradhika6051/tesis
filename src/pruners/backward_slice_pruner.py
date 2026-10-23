from dataset.pruners.base_pruner import BasePruner
from src.pruners.utils import prune_cfg_preserving_virtual_paths

from src.pruners.cfg_traversal import traverse_cfg

class BackwardSlicePruner(BasePruner):

    def prune(self, sample):
        reachable = traverse_cfg(
            cfg=sample.cfg,
            seed_nodes=sample.seed_nodes,
            function_nodes=sample.function_nodes,
            direction="backward",
        )

        sample.pruned_cfg = prune_cfg_preserving_virtual_paths(
            sample.cfg,
            reachable,
        )

        return sample