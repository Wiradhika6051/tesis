from tesis.dataset.pruners.base_pruner import BasePruner
from tesis.dataset.pruners.utils import prune_cfg


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