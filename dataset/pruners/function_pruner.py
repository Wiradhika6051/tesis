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

        target_nodes = cfg.get(
            "target_nodes",
            []
        )

        if not target_nodes:
            return {
                **cfg,
                "kept_lines": {
                    n.lineno
                    for n in cfg["nodes"]
                }
            }

        function_range = cfg.get(
            "function_ranges",
            {}
        )

        keep = set()

        for target in target_nodes:

            for (
                start,
                end
            ) in function_range.values():

                node = cfg["id_to_node"][
                    target
                ]

                if (
                    start
                    <=
                    node.lineno
                    <=
                    end
                ):

                    for n in cfg["nodes"]:

                        if (
                            start
                            <=
                            n.lineno
                            <=
                            end
                        ):
                            keep.add(
                                n.node_id
                            )

        return prune_cfg(
            cfg,
            keep
        )