from src.localizer.CFGLocalizer import CFGLocalizer


class LineCFGLocalizer(CFGLocalizer):
    def localize(self, sample):
        sample.line_to_node = {}
        sample.seed_nodes = []
        sample.unmatched_seed_lines = []
        sample.localization_status = None
        sample.exclusion_reason = None

        seed_lines = list(dict.fromkeys(sample.seed_lines or []))

        # No changed lines is a distinct condition, not a mapping failure.
        if not seed_lines:
            sample.localization_status = "no_changed_lines"
            sample.exclusion_reason = "no_changed_lines"
            return sample

        nodes = sample.cfg["nodes"]

        for line in seed_lines:
            matched_node_ids = []

            for node in nodes:
                start = node.lineno
                end = node.end_lineno

                # Ignore virtual nodes and invalid source spans.
                if start < 1 or end < start:
                    continue

                # Retain every valid node whose span contains this line.
                if start <= line <= end:
                    matched_node_ids.append(node.node_id)

            sample.line_to_node[line] = matched_node_ids

            if matched_node_ids:
                sample.seed_nodes.extend(matched_node_ids)
            else:
                sample.unmatched_seed_lines.append(line)

        # Deduplicate node IDs without changing their first-seen order.
        sample.seed_nodes = list(dict.fromkeys(sample.seed_nodes))

        if not sample.seed_nodes:
            sample.localization_status = "failed"
            sample.exclusion_reason = "no_valid_seed_nodes"
        elif sample.unmatched_seed_lines:
            sample.localization_status = "partial"
        else:
            sample.localization_status = "success"

        return sample