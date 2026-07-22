from tesis.dataset.pruners.base_pruner import BasePruner
from tesis.dataset.pruners.utils import prune_cfg


class IdentityPruner(BasePruner):

    def prune(
        self,
        cfg,
        start_nodes
    ):

        keep = {
            node.node_id
            for node in cfg["nodes"]
        }

        return prune_cfg(
            cfg,
            keep
        )