from src.localizer.CFGLocalizer import CFGLocalizer


class FunctionScopeLocalizer(CFGLocalizer):

    def localize(self, sample):
        sample.function_nodes = []
        sample.function_scope_status = "failed"
        sample.function_scope_reason = None

        cfg = getattr(sample, "cfg", None)

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

        # 1. Prefer exact eligible lines from FunctionSampleBuilder.
        body_lines = getattr(sample, "function_body_lines", None)

        if body_lines is not None:
            try:
                eligible_lines = {
                    line for line in body_lines
                    if isinstance(line, int) and line > 0
                }
            except TypeError:
                sample.function_scope_reason = "invalid_function_body_lines"
                sample.seed_nodes = []
                return sample

            if not eligible_lines:
                sample.function_scope_reason = "empty_function_body_lines"
                sample.seed_nodes = []
                return sample

        else:
            # 2. Support one or more explicit ranges.
            scopes = getattr(sample, "function_scopes", None)

            if scopes is not None:
                if not scopes:
                    sample.function_scope_reason = "no_function_scopes"
                    sample.seed_nodes = []
                    return sample

                eligible_lines = set()

                for scope in scopes:
                    if not isinstance(scope, dict):
                        sample.function_scope_reason = "invalid_function_range"
                        sample.seed_nodes = []
                        return sample

                    start = scope.get("start")
                    end = scope.get("end")

                    if (
                        not isinstance(start, int)
                        or not isinstance(end, int)
                        or start <= 0
                        or end < start
                    ):
                        sample.function_scope_reason = "invalid_function_range"
                        sample.seed_nodes = []
                        return sample

                    eligible_lines.update(range(start, end + 1))

            else:
                # 3. Backward-compatible single-range fallback.
                start = getattr(sample, "function_start", None)
                end = getattr(sample, "function_end", None)

                if start is None or end is None:
                    sample.function_scope_reason = "no_function_scopes"
                    sample.seed_nodes = []
                    return sample

                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or start <= 0
                    or end < start
                ):
                    sample.function_scope_reason = "invalid_function_range"
                    sample.seed_nodes = []
                    return sample

                eligible_lines = set(range(start, end + 1))


        function_nodes = []
        seen_node_ids = set()

        for node in nodes:
            if isinstance(node, dict):
                lineno = node.get("lineno")
                node_id = node.get("node_id")
            else:
                lineno = getattr(node, "lineno", None)
                node_id = getattr(node, "node_id", None)

            if not isinstance(lineno, int) or lineno <= 0:
                continue

            if node_id is None or lineno not in eligible_lines:
                continue

            if node_id in seen_node_ids:
                continue

            seen_node_ids.add(node_id)
            function_nodes.append(node_id)


        sample.function_nodes = function_nodes

        seed_nodes = set(getattr(sample, "seed_nodes", None) or [])
        valid_seeds = seed_nodes.intersection(function_nodes)
        sample.seed_nodes = sorted(valid_seeds)

        if not function_nodes:
            sample.function_scope_reason = "no_nodes_in_function_scope"
            return sample

        if not valid_seeds:
            sample.function_scope_reason = "no_seed_nodes_in_function_scope"
            return sample

        sample.function_scope_status = "success"
        return sample
