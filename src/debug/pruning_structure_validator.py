class PruningStructureValidator:
    """
    Validates structural correctness of a pruned CFG.

    This validator checks:
      1. Mapping correctness
      2. Seed preservation
      3. Function-scope correctness
      4. Edge correctness
      5. Node-content preservation
      6. Pruned CFG structural integrity
    """

    def validate(self, record):
        checks = {
            "mapping": self._validate_mapping(record),
            "seed_preservation": self._validate_seed_preservation(record),
            "function_scope": self._validate_function_scope(record),
            "edges": self._validate_edges(record),
            "node_content": self._validate_node_content(record),
            "graph_integrity": self._validate_graph_integrity(record),
        }

        valid = all(result["valid"] for result in checks.values())

        failed_checks = [
            name
            for name, result in checks.items()
            if not result["valid"]
        ]

        return {
            "sample_id": record.get("sample_id"),
            "valid": valid,
            "checks": checks,
            "failed_checks": failed_checks,
        }

    # ============================================================
    # 1. Mapping correctness
    # ============================================================

    def _validate_mapping(self, record):
        pruned_cfg = record["pruned_cfg"]

        mapping = pruned_cfg.get("original_to_pruned")

        if mapping is None:
            return self._fail(
                "Missing original_to_pruned mapping"
            )

        pruned_nodes = pruned_cfg.get("nodes", [])

        pruned_ids = {
            node["node_id"]
            for node in pruned_nodes
        }

        mapped_ids = {
            int(pruned_id)
            for pruned_id in mapping.values()
        }

        # Every pruned node must have exactly one mapping.
        if len(mapping) != len(pruned_nodes):
            return self._fail(
                "Mapping size ({}) != pruned node count ({})".format(
                    len(mapping),
                    len(pruned_nodes)
                )
            )

        # Mapping must be one-to-one.
        if len(mapped_ids) != len(mapping):
            return self._fail(
                "original_to_pruned contains duplicate pruned IDs"
            )

        # Mapping must cover every pruned node.
        if mapped_ids != pruned_ids:
            return self._fail(
                "Mapping does not cover exactly the pruned node IDs"
            )

        return self._ok()

    # ============================================================
    # 2. Seed preservation
    # ============================================================

    def _validate_seed_preservation(self, record):
        mapping = record["pruned_cfg"].get(
            "original_to_pruned",
            {}
        )

        retained_original_nodes = {
            int(original_id)
            for original_id in mapping.keys()
        }

        seed_nodes = set(
            record.get("seed_nodes", [])
        )

        missing_seeds = (
            seed_nodes - retained_original_nodes
        )

        if missing_seeds:
            return self._fail(
                "Seed nodes removed by pruning: {}".format(
                    sorted(missing_seeds)
                )
            )

        return self._ok()

    # ============================================================
    # 3. Function-scope correctness
    # ============================================================

    def _validate_function_scope(self, record):
        mapping = record["pruned_cfg"].get(
            "original_to_pruned",
            {}
        )

        retained_original_nodes = {
            int(original_id)
            for original_id in mapping.keys()
        }

        function_nodes = set(
            record.get("function_nodes", [])
        )

        outside_function = (
            retained_original_nodes - function_nodes
        )

        if outside_function:
            return self._fail(
                "Nodes outside function scope were retained: {}".format(
                    sorted(outside_function)
                )
            )

        return self._ok()

    # ============================================================
    # 4. Edge correctness
    # ============================================================

    def _validate_edges(self, record):
        original_cfg = record["cfg"]
        pruned_cfg = record["pruned_cfg"]

        original_edges = {
            tuple(edge)
            for edge in original_cfg.get("edges", [])
        }

        mapping = pruned_cfg.get(
            "original_to_pruned",
            {}
        )

        # Convert:
        #
        # original -> pruned
        #
        # into:
        #
        # pruned -> original
        reverse_mapping = {
            int(pruned_id): int(original_id)
            for original_id, pruned_id in mapping.items()
        }

        converted_edges = set()

        for edge in pruned_cfg.get("edges", []):
            if len(edge) != 2:
                return self._fail(
                    "Invalid pruned edge: {}".format(edge)
                )

            src, dst = edge

            if src not in reverse_mapping:
                return self._fail(
                    "Pruned edge source {} has no mapping".format(src)
                )

            if dst not in reverse_mapping:
                return self._fail(
                    "Pruned edge destination {} has no mapping".format(dst)
                )

            original_src = reverse_mapping[src]
            original_dst = reverse_mapping[dst]

            converted_edges.add(
                (original_src, original_dst)
            )

        # Every retained edge must have existed
        # in the original CFG.
        invalid_edges = (
            converted_edges - original_edges
        )

        if invalid_edges:
            return self._fail(
                "Pruned CFG contains edges not present "
                "in original CFG: {}".format(
                    sorted(invalid_edges)
                )
            )

        return self._ok()

    # ============================================================
    # 5. Node-content preservation
    # ============================================================

    def _validate_node_content(self, record):
        original_nodes = {
            node["node_id"]: node
            for node in record["cfg"].get("nodes", [])
        }

        pruned_nodes = {
            node["node_id"]: node
            for node in record["pruned_cfg"].get("nodes", [])
        }

        mapping = record["pruned_cfg"].get(
            "original_to_pruned",
            {}
        )

        for original_id, pruned_id in mapping.items():
            original_id = int(original_id)
            pruned_id = int(pruned_id)

            if original_id not in original_nodes:
                return self._fail(
                    "Mapped original node {} does not exist".format(
                        original_id
                    )
                )

            if pruned_id not in pruned_nodes:
                return self._fail(
                    "Mapped pruned node {} does not exist".format(
                        pruned_id
                    )
                )

            original = original_nodes[original_id]
            pruned = pruned_nodes[pruned_id]

            fields = [
                "lineno",
                "end_lineno",
                "node_type",
                "text",
            ]

            for field in fields:
                if original.get(field) != pruned.get(field):
                    return self._fail(
                        "Node content mismatch for "
                        "original {} -> pruned {}: field '{}'".format(
                            original_id,
                            pruned_id,
                            field
                        )
                    )

        return self._ok()

    # ============================================================
    # 6. Graph integrity
    # ============================================================

    def _validate_graph_integrity(self, record):
        pruned_cfg = record["pruned_cfg"]

        node_ids = {
            node["node_id"]
            for node in pruned_cfg.get("nodes", [])
        }

        for src, dst in pruned_cfg.get("edges", []):
            if src not in node_ids:
                return self._fail(
                    "Edge source {} does not exist "
                    "in pruned nodes".format(src)
                )

            if dst not in node_ids:
                return self._fail(
                    "Edge destination {} does not exist "
                    "in pruned nodes".format(dst)
                )

        return self._ok()

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _ok():
        return {
            "valid": True,
            "reason": "OK",
        }

    @staticmethod
    def _fail(reason):
        return {
            "valid": False,
            "reason": reason,
        }