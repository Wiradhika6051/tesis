class PruningValidator:

    def validate(self, record):
        raise NotImplementedError(
            "PruningValidator.validate() must be implemented"
        )

    @staticmethod
    def build_result(
        record,
        valid,
        reason,
        expected_nodes,
        retained_nodes,
        missing_nodes=None,
        unexpected_nodes=None,
        missing_seeds=None
    ):
        expected_nodes = set(expected_nodes)
        retained_nodes = set(retained_nodes)

        if missing_nodes is None:
            missing_nodes = expected_nodes - retained_nodes

        if unexpected_nodes is None:
            unexpected_nodes = retained_nodes - expected_nodes

        if missing_seeds is None:
            seed_nodes = set(record.get("seed_nodes", []))
            missing_seeds = seed_nodes - retained_nodes

        total_nodes = len(
            record.get("cfg", {}).get("nodes", [])
        )

        retention_ratio = (
            float(len(retained_nodes)) / total_nodes
            if total_nodes > 0
            else 0.0
        )

        return {
            "sample_id": record.get("sample_id", ""),
            "repo": record.get("repo", ""),
            "file_path": record.get("file_path", ""),
            "commit": record.get("commit", ""),
            "label": record.get("label"),

            "valid": bool(valid),
            "reason": reason,

            "expected_nodes": sorted(expected_nodes),
            "retained_nodes": sorted(retained_nodes),

            "missing_nodes": sorted(missing_nodes),
            "unexpected_nodes": sorted(unexpected_nodes),
            "missing_seeds": sorted(missing_seeds),

            "retention_ratio": retention_ratio
        }