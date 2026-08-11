from src.pruners.base_pruner import BasePruner
from src.pruners.utils import prune_cfg


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
        sample.pruned_cfg = prune_cfg(
            sample.cfg,
            keep
        )

        return sample