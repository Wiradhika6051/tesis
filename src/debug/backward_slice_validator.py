from collections import deque


class BackwardSliceValidator:

    def validate(self, sample):

        if not sample.cfg:
            return self._failure(
                "missing_cfg"
            )

        if not sample.pruned_cfg:
            return self._failure(
                "missing_pruned_cfg"
            )

        cfg = sample.cfg

        all_nodes = {
            node.node_id
            for node in cfg.get("nodes", [])
        }

        original_to_pruned = sample.pruned_cfg.get(
            "original_to_pruned",
            {}
        )
        
        retained_nodes = {
            int(original_id)
            for original_id in original_to_pruned.keys()
        }

        seed_nodes = set(
            sample.seed_nodes or []
        )

        function_nodes = set(
            sample.function_nodes or []
        )

        # --------------------------------------------------
        # 1. Validate seed nodes themselves
        # --------------------------------------------------

        missing_seed_nodes = (
            seed_nodes - all_nodes
        )

        if missing_seed_nodes:
            return self._failure(
                "seed_nodes_not_in_cfg",
                missing_seed_nodes=missing_seed_nodes
            )

        # --------------------------------------------------
        # 2. Every valid seed should be retained
        # --------------------------------------------------

        valid_seeds = (
            seed_nodes & function_nodes
        )

        missing_seeds = (
            valid_seeds - retained_nodes
        )

        # --------------------------------------------------
        # 3. Compute expected backward slice
        # --------------------------------------------------

        expected_nodes = self._compute_backward_slice(
            cfg=cfg,
            seeds=valid_seeds,
            function_nodes=function_nodes
        )

        # --------------------------------------------------
        # 4. Compare expected vs actual
        # --------------------------------------------------

        unexpected_nodes = (
            retained_nodes - expected_nodes
        )

        missing_nodes = (
            expected_nodes - retained_nodes
        )

        is_valid = (
            not missing_seeds
            and not unexpected_nodes
            and not missing_nodes
        )

        return {
            "valid": is_valid,

            "reason": (
                "valid"
                if is_valid
                else "slice_mismatch"
            ),

            "seed_count": len(seed_nodes),

            "valid_seed_count": len(valid_seeds),

            "retained_count": len(
                retained_nodes
            ),

            "expected_count": len(
                expected_nodes
            ),

            "missing_seeds": sorted(
                missing_seeds
            ),

            "unexpected_nodes": sorted(
                unexpected_nodes
            ),

            "missing_nodes": sorted(
                missing_nodes
            ),

            "retention_ratio": (
                float(len(retained_nodes))
                / len(all_nodes)
                if all_nodes
                else 0.0
            )
        }

    @staticmethod
    def _compute_backward_slice(
        cfg,
        seeds,
        function_nodes
    ):

        # --------------------------------------------------
        # Build reverse CFG
        #
        # Original edge:
        #
        #     A -> B
        #
        # Reverse traversal:
        #
        #     B -> A
        # --------------------------------------------------

        reverse_graph = {}

        for src, dst in cfg.get(
            "edges",
            []
        ):

            reverse_graph.setdefault(
                dst,
                []
            ).append(src)

        # --------------------------------------------------
        # Traverse backwards from seed nodes.
        # --------------------------------------------------

        expected = set(seeds)

        queue = deque(seeds)

        while queue:

            node = queue.popleft()

            for parent in reverse_graph.get(
                node,
                []
            ):

                # Only retain nodes belonging
                # to the target function.
                if parent not in function_nodes:
                    continue

                if parent in expected:
                    continue

                expected.add(parent)
                queue.append(parent)

        return expected

    @staticmethod
    def _failure(reason, **details):

        result = {
            "valid": False,
            "reason": reason
        }

        result.update(details)

        return result