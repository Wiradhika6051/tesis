
from src.localizer.CFGLocalizer import CFGLocalizer


class LineCFGLocalizer(CFGLocalizer):

    def localize(self, sample):
        sample.line_to_node = {}
        sample.seed_nodes = []

        for line in sample.seed_lines:
            matched_nodes = []

            for node in sample.cfg["nodes"]:
                start = node.lineno
                end = node.end_lineno

                # Ignore virtual CFG nodes.
                if start < 1 or end < start:
                    continue

                if start <= line <= end:
                    matched_nodes.append(node)

            # Prefer the smallest matching source span.
            matched_nodes.sort(
                key=lambda node: (
                    node.end_lineno - node.lineno,
                    node.node_id,
                )
            )

            selected_nodes = (
                [matched_nodes[0].node_id]
                if matched_nodes
                else []
            )

            sample.line_to_node[line] = selected_nodes
            sample.seed_nodes.extend(selected_nodes)

        sample.seed_nodes = list(
            dict.fromkeys(sample.seed_nodes)
        )

        return sample
