from collections import deque

from src.pruners.base_pruner import BasePruner
from src.pruners.utils import prune_cfg


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

        function_nodes = set(
            sample.function_nodes
        )

        valid_seeds = (
            set(seed_nodes)
            &
            set(function_nodes)
        )
        if not valid_seeds:
            print(
                "[BACKWARD DEBUG]",
                "seed_nodes =", seed_nodes,
                "function_nodes =", function_nodes,
                "cfg_nodes =", len(cfg["nodes"])
            )
        
        keep = set(
            valid_seeds
        )
        
        queue = deque(
            valid_seeds
        )

        while queue:
            node = queue.popleft()

            for parent in reverse_graph.get(node, []):
                #
                # Stay inside the function.
                #
                if parent not in function_nodes:
                    continue
                
                if parent in keep:
                    continue
            
                keep.add(parent)
                queue.append(parent)

        sample.pruned_cfg = prune_cfg(
            cfg,
            keep
        )

        if len(sample.pruned_cfg["nodes"]) == 0:
            print(
                "[BACKWARD EMPTY]",
                "seed_nodes =", seed_nodes,
                "function_nodes =", function_nodes,
                "valid_seeds =", valid_seeds,
                "cfg_nodes =", len(cfg["nodes"])
            )

        return sample