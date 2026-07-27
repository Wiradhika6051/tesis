from tesis.dataset.pruners.base_pruner import BasePruner
from tesis.dataset.pruners.utils import prune_cfg


class IdentityPruner(BasePruner):

    def prune(
        self,
        sample
    ):

        keep = {
            node.node_id
            for node in sample.cfg["nodes"]
        }

        sample.pruned_cfg = prune_cfg(
            sample.cfg,
            keep
        )

        return sample