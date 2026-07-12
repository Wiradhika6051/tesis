from tesis.dataset.pruners.base_pruner import BasePruner
from tesis.dataset.pruners.utils import prune_cfg


class FunctionPruner(
    BasePruner
):

    def prune(
        self,
        cfg,
        snippet
    ):

        return {
            **cfg,
            "kept_lines": {
                node.lineno
                for node in cfg["nodes"]
            }
        }