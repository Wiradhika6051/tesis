class LineCFGLocalizer(CFGLocalizer):

    def localize(
        self,
        sample
    ):

        mapping = {}

        for node in sample.cfg["nodes"]:

            mapping.setdefault(
                node.lineno,
                []
            ).append(
                node.node_id
            )

        sample.line_to_node = mapping

        sample.seed_nodes = []

        for line in sample.seed_lines:

            sample.seed_nodes.extend(
                mapping.get(line, [])
            )

        return sample