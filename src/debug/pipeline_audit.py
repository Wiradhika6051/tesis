import json
import os
from dataclasses import asdict, is_dataclass


class PipelineAuditExporter:

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # Public API
    # ============================================================

    def export_raw_samples(self, samples):
        self._export_jsonl(
            "01_raw_samples.jsonl",
            [
                self._serialize_raw_sample(sample)
                for sample in samples
            ]
        )

    def export_diff_localized(self, samples):
        self._export_jsonl(
            "02_diff_localized.jsonl",
            [
                self._serialize_diff_sample(sample)
                for sample in samples
                if getattr(sample, "seed_lines", None)
            ]
        )

    def export_function_samples(self, samples):
        self._export_jsonl(
            "03_function_samples.jsonl",
            [
                self._serialize_function_sample(sample)
                for sample in samples
            ]
        )

    def export_cfg(self, samples):
        self._export_jsonl(
            "04_cfg.jsonl",
            [
                self._serialize_cfg_sample(sample)
                for sample in samples
                if getattr(sample, "cfg", None)
            ]
        )

    def export_localized(self, samples):
        self._export_jsonl(
            "05_seed_nodes.jsonl",
            [
                self._serialize_localized_sample(sample)
                for sample in samples
                if getattr(sample, "cfg", None)
            ]
        )

    def export_function_scope(self, samples):
        self._export_jsonl(
            "06_function_scope.jsonl",
            [
                self._serialize_function_scope(sample)
                for sample in samples
            ]
        )

    def export_pruned(self, samples, strategy_name):
        filename = "07_pruned_cfg_{}.jsonl".format(
            self._safe_name(strategy_name)
        )

        self._export_jsonl(
            filename,
            [
                self._serialize_pruned_sample(sample)
                for sample in samples
                if getattr(sample, "pruned_cfg", None)
            ]
        )

    def export_tokens(self, samples):
        self._export_jsonl(
            "08_tokens.jsonl",
            [
                self._serialize_tokens(sample)
                for sample in samples
                if hasattr(sample, "tokens")
            ]
        )

    def export_encoded(self, samples):
        self._export_jsonl(
            "09_encoded_samples.jsonl",
            [
                self._serialize_encoded(sample)
                for sample in samples
            ]
        )

    # ============================================================
    # Serialization
    # ============================================================

    def _serialize_raw_sample(self, sample):
        return {
            "sample_id": self._sample_id(sample),
            "repo": getattr(sample, "repo", ""),
            "parent_commit": getattr(sample, "parent_commit", ""),
            "commit": getattr(sample, "commit", ""),
            "file_path": getattr(sample, "file_path", ""),
            "label": getattr(sample, "label", None),
            "source": getattr(sample, "source", ""),
            "diff": getattr(sample, "diff", ""),
        }

    def _serialize_diff_sample(self, sample):
        return {
            "sample_id": self._sample_id(sample),
            "repo": getattr(sample, "repo", ""),
            "parent_commit": getattr(sample, "parent_commit", ""),
            "commit": getattr(sample, "commit", ""),
            "file_path": getattr(sample, "file_path", ""),
            "label": getattr(sample, "label", None),
            "seed_lines": list(
                getattr(sample, "seed_lines", [])
            ),
        }

    def _serialize_function_sample(self, sample):
        return {
            "sample_id": self._sample_id(sample),
            "repo": getattr(sample, "repo", ""),
            "parent_commit": getattr(sample, "parent_commit", ""),
            "commit": getattr(sample, "commit", ""),
            "file_path": getattr(sample, "file_path", ""),
            "label": getattr(sample, "label", None),

            "function_name": getattr(
                sample,
                "function_name",
                None
            ),

            "function_start": getattr(
                sample,
                "function_start",
                None
            ),

            "function_end": getattr(
                sample,
                "function_end",
                None
            ),

            "seed_lines": list(
                getattr(sample, "seed_lines", [])
            ),
        }

    def _serialize_cfg_sample(self, sample):
        cfg = sample.cfg

        return {
            "sample_id": self._sample_id(sample),
            "function_name": getattr(
                sample,
                "function_name",
                None
            ),
            "function_start": getattr(
                sample,
                "function_start",
                None
            ),
            "function_end": getattr(
                sample,
                "function_end",
                None
            ),
            "nodes": self._serialize_nodes(cfg),
            "edges": self._serialize_edges(cfg),
        }

    def _serialize_localized_sample(self, sample):
        return {
            "sample_id": self._sample_id(sample),
            "seed_lines": list(
                getattr(sample, "seed_lines", [])
            ),
            "seed_nodes": list(
                getattr(sample, "seed_nodes", [])
            ),
            "line_to_node": getattr(
                sample,
                "line_to_node",
                {}
            ),
        }

    def _serialize_function_scope(self, sample):
        return {
            "sample_id": self._sample_id(sample),
            "function_name": getattr(
                sample,
                "function_name",
                None
            ),
            "function_start": getattr(
                sample,
                "function_start",
                None
            ),
            "function_end": getattr(
                sample,
                "function_end",
                None
            ),
            "function_nodes": list(
                getattr(sample, "function_nodes", [])
            ),
            "seed_nodes": list(
                getattr(sample, "seed_nodes", [])
            ),
        }

    def _serialize_pruned_sample(self, sample):
        cfg = sample.pruned_cfg

        return {
            "sample_id": self._sample_id(sample),
            "seed_lines": list(
                getattr(sample, "seed_lines", [])
            ),
            "seed_nodes": list(
                getattr(sample, "seed_nodes", [])
            ),
            "function_nodes": list(
                getattr(sample, "function_nodes", [])
            ),
            "nodes": self._serialize_nodes(cfg),
            "edges": self._serialize_edges(cfg),
            "original_to_pruned": {
                str(original): pruned
                for original, pruned in cfg.get(
                    "original_to_pruned",
                    {}
                ).items()
            },
        }

    def _serialize_tokens(self, sample):
        return {
            "sample_id": self._sample_id(sample),
            "tokens": list(
                getattr(sample, "tokens", [])
            ),
        }

    def _serialize_encoded(self, sample):
        result = {
            "sample_id": self._sample_id(sample),
        }

        # Don't dump huge tensors into JSON by default.
        if hasattr(sample, "encoded"):
            encoded = sample.encoded

            try:
                result["shape"] = list(encoded.shape)
                result["dtype"] = str(encoded.dtype)
            except AttributeError:
                result["type"] = type(encoded).__name__

        if hasattr(sample, "node_features"):
            features = sample.node_features

            try:
                result["node_feature_shape"] = list(
                    features.shape
                )
            except AttributeError:
                pass

        return result

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _serialize_nodes(cfg):
        result = []

        for node in cfg.get("nodes", []):
            if isinstance(node, dict):
                result.append(node)
            else:
                result.append({
                    "node_id": node.node_id,
                    "lineno": node.lineno,
                    "end_lineno": node.end_lineno,
                    "node_type": node.node_type,
                    "text": node.text,
                })

        return result

    @staticmethod
    def _serialize_edges(cfg):
        return [
            list(edge)
            for edge in cfg.get("edges", [])
        ]

    @staticmethod
    def _sample_id(sample):
        repo = getattr(sample, "repo", "")
        commit = getattr(
            sample,
            "parent_commit",
            getattr(sample, "commit", "")
        )
        file_path = getattr(sample, "file_path", "")
        function_start = getattr(
            sample,
            "function_start",
            None
        )
        function_end = getattr(
            sample,
            "function_end",
            None
        )
        label = getattr(sample, "label", None)

        return "{}:{}:{}:{}:{}:{}".format(
            repo,
            commit,
            file_path,
            function_start,
            function_end,
            label,
        )

    @staticmethod
    def _safe_name(name):
        return (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )

    def _export_jsonl(self, filename, records):
        path = os.path.join(
            self.output_dir,
            filename
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            for record in records:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

        print(
            "[PIPELINE AUDIT] {} records -> {}".format(
                len(records),
                path
            )
        )

        return path