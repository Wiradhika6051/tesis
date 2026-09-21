from src.localizer.CFGLocalizer import CFGLocalizer


class FunctionScopeLocalizer(CFGLocalizer):

    def localize(self, sample):

        start = sample.function_start
        end = sample.function_end

        sample.function_nodes = [
            node.node_id
            for node in sample.cfg["nodes"]
            if start <= node.lineno <= end
        ]

        return sample