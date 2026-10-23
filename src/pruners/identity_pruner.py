from src.pruners.base_pruner import BasePruner
from src.pruners.utils import prune_cfg_preserving_virtual_paths


class IdentityPruner(BasePruner):

    def prune(
        self,
        sample
    ):

        #
        # Keep the entire localized function.
        #
        keep = set(
            sample.function_nodes
        )

        #
        # Build the pruned CFG.
        #
        sample.pruned_cfg = prune_cfg_preserving_virtual_paths(
            sample.cfg,
            keep
        )

        return sample