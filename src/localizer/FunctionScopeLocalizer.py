from src.localizer.CFGLocalizer import CFGLocalizer



class FunctionScopeLocalizer(CFGLocalizer):

    FUNCTION_DEFINITION_TYPES = {
        "FunctionDef",
        "AsyncFunctionDef",
    }

    def localize(self, sample):
        sample.function_nodes = []
        sample.function_scope_status = "failed"
        sample.function_scope_reason = None

        cfg = getattr(sample, "cfg", None)

        # 1. Validate the CFG.
        if not isinstance(cfg, dict):
            sample.function_scope_reason = "missing_or_invalid_cfg"
            sample.seed_nodes = []
            return sample

        nodes = cfg.get("nodes")

        if not isinstance(nodes, list):
            sample.function_scope_reason = "missing_or_invalid_cfg_nodes"
            sample.seed_nodes = []
            return sample

        if not nodes:
            sample.function_scope_reason = "empty_cfg"
            sample.seed_nodes = []
            return sample

        # 2. Determine the eligible source lines.
        body_lines = getattr(sample, "function_body_lines", None)

        if body_lines is not None:
            try:
                eligible_lines = {
                    line
                    for line in body_lines
                    if isinstance(line, int)
                    and not isinstance(line, bool)
                    and line > 0
                }
            except TypeError:
                sample.function_scope_reason = (
                    "invalid_function_body_lines"
                )
                sample.seed_nodes = []
                return sample

            if not eligible_lines:
                sample.function_scope_reason = (
                    "empty_function_body_lines"
                )
                sample.seed_nodes = []
                return sample

        else:
            # Backward-compatible support for explicit scopes.
            scopes = getattr(sample, "function_scopes", None)

            if scopes is not None:
                if not scopes:
                    sample.function_scope_reason = (
                        "no_function_scopes"
                    )
                    sample.seed_nodes = []
                    return sample

                eligible_lines = set()

                for scope in scopes:
                    if not isinstance(scope, dict):
                        sample.function_scope_reason = (
                            "invalid_function_range"
                        )
                        sample.seed_nodes = []
                        return sample

                    start = scope.get("start")
                    end = scope.get("end")

                    if (
                        not isinstance(start, int)
                        or isinstance(start, bool)
                        or not isinstance(end, int)
                        or isinstance(end, bool)
                        or start <= 0
                        or end < start
                    ):
                        sample.function_scope_reason = (
                            "invalid_function_range"
                        )
                        sample.seed_nodes = []
                        return sample

                    eligible_lines.update(range(start, end + 1))

            else:
                # Backward-compatible single-range fallback.
                start = getattr(sample, "function_start", None)
                end = getattr(sample, "function_end", None)

                if start is None or end is None:
                    sample.function_scope_reason = (
                        "no_function_scopes"
                    )
                    sample.seed_nodes = []
                    return sample

                if (
                    not isinstance(start, int)
                    or isinstance(start, bool)
                    or not isinstance(end, int)
                    or isinstance(end, bool)
                    or start <= 0
                    or end < start
                ):
                    sample.function_scope_reason = (
                        "invalid_function_range"
                    )
                    sample.seed_nodes = []
                    return sample

                eligible_lines = set(range(start, end + 1))

        # 3. Collect nodes belonging to the function body.
        function_nodes = []
        seen_node_ids = set()

        for node in nodes:
            if isinstance(node, dict):
                node_id = node.get("node_id")
                lineno = node.get("lineno")
                node_type = node.get("node_type")
            else:
                node_id = getattr(node, "node_id", None)
                lineno = getattr(node, "lineno", None)
                node_type = getattr(node, "node_type", None)

            if node_id is None:
                continue

            if not isinstance(lineno, int) or isinstance(lineno, bool):
                continue

            if lineno <= 0 or lineno not in eligible_lines:
                continue

            # A function definition is not a body statement.
            if node_type in self.FUNCTION_DEFINITION_TYPES:
                continue

            if node_id in seen_node_ids:
                continue

            seen_node_ids.add(node_id)
            function_nodes.append(node_id)

        sample.function_nodes = function_nodes
        function_node_ids = set(function_nodes)

        # 4. Keep only seeds that belong to the eligible function body.
        original_seeds = set(
            getattr(sample, "seed_nodes", None) or []
        )

        sample.seed_nodes = sorted(
            original_seeds.intersection(function_node_ids)
        )

        # 5. Keep line_to_node consistent with surviving seeds.
        valid_seed_ids = set(sample.seed_nodes)

        line_to_node = getattr(sample, "line_to_node", None)

        if isinstance(line_to_node, dict):
            sample.line_to_node = {
                line: [
                    node_id
                    for node_id in (node_ids or [])
                    if node_id in valid_seed_ids
                ]
                for line, node_ids in line_to_node.items()
            }

        # 6. Recalculate localization status after scope filtering.
        if not function_nodes:
            sample.function_scope_reason = (
                "no_nodes_in_function_scope"
            )
            sample.localization_status = "failed"
            sample.exclusion_reason = "no_valid_seed_nodes"
            return sample

        if not sample.seed_nodes:
            sample.function_scope_reason = (
                "no_seed_nodes_in_function_scope"
            )
            sample.localization_status = "failed"
            sample.exclusion_reason = "no_valid_seed_nodes"
            return sample

        sample.function_scope_status = "success"
        sample.function_scope_reason = None

        if (
            getattr(sample, "unmatched_seed_lines", None)
            or getattr(sample, "container_only_seed_lines", None)
        ):
            sample.localization_status = "partial"
        else:
            sample.localization_status = "success"

        sample.exclusion_reason = None
        return sample
