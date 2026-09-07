from src.localizer.CFGLocalizer import CFGLocalizer


class LineCFGLocalizer(CFGLocalizer):

    def localize(
        self,
        sample
    ):

        sample.line_to_node = {}

        sample.seed_nodes = []

        for line in sample.seed_lines:

            matched_nodes = []

            for node in sample.cfg["nodes"]:

                if (
                    node.lineno
                    <= line
                    <= node.end_lineno
                ):

                    matched_nodes.append(
                        node.node_id
                    )

            sample.line_to_node[
                line
            ] = matched_nodes

            sample.seed_nodes.extend(
                matched_nodes
            )

        return sample