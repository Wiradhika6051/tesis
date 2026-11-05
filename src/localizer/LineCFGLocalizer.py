from src.localizer.CFGLocalizer import CFGLocalizer


class LineCFGLocalizer(CFGLocalizer):

    FUNCTION_DEFINITION_TYPES = {
        "FunctionDef",
        "AsyncFunctionDef",
    }

    def localize(self, sample):
        sample.line_to_node = {}
        sample.seed_nodes = []
        sample.unmatched_seed_lines = []
        sample.container_only_seed_lines = []
        sample.localization_status = None
        sample.exclusion_reason = None

        seed_lines = list(dict.fromkeys(sample.seed_lines or []))

        if not seed_lines:
            sample.localization_status = "no_changed_lines"
            sample.exclusion_reason = "no_changed_lines"
            return sample

        nodes = sample.cfg["nodes"]

        for line in seed_lines:
            matching_nodes = []
            definition_nodes = []

            for node in nodes:
                start = node.lineno
                end = node.end_lineno
                node_type = node.node_type

                if start < 1 or end < start:
                    continue

                if not (start <= line <= end):
                    continue

                if node_type in self.FUNCTION_DEFINITION_TYPES:
                    definition_nodes.append(node)
                    continue

                matching_nodes.append(node)

            # Do not use a function definition as a fallback seed.
            if not matching_nodes:
                sample.line_to_node[line] = []

                if definition_nodes:
                    sample.container_only_seed_lines.append(line)
                else:
                    sample.unmatched_seed_lines.append(line)

                continue

            # Prefer the smallest source span.
            # Keep ties because multiple CFG nodes can legitimately
            # represent different AST constructs on the same line.
            min_span = min(
                node.end_lineno - node.lineno
                for node in matching_nodes
            )

            selected_nodes = [
                node
                for node in matching_nodes
                if node.end_lineno - node.lineno == min_span
            ]

            selected_ids = list(dict.fromkeys(
                node.node_id for node in selected_nodes
            ))

            sample.line_to_node[line] = selected_ids
            sample.seed_nodes.extend(selected_ids)

        sample.seed_nodes = list(dict.fromkeys(sample.seed_nodes))

        if not sample.seed_nodes:
            sample.localization_status = "failed"
            sample.exclusion_reason = "no_valid_seed_nodes"

        elif (
            sample.unmatched_seed_lines
            or sample.container_only_seed_lines
        ):
            sample.localization_status = "partial"

        else:
            sample.localization_status = "success"

        return sample