import json
import os


class PruningAuditExporter:

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export(self, samples, pruner_name):
        filename = "{}.jsonl".format(
            self._safe_name(pruner_name)
        )

        output_path = os.path.join(
            self.output_dir,
            filename
        )

        count = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:

                if not getattr(sample, "cfg", None):
                    continue

                if not getattr(sample, "pruned_cfg", None):
                    continue

                record = self._build_record(sample)

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

                count += 1

        print(
            "[PRUNING AUDIT] Exported {} samples -> {}".format(
                count,
                output_path
            )
        )

        return output_path

    def _build_record(self, sample):
        cfg = sample.cfg
        pruned_cfg = sample.pruned_cfg

        return {
            "sample_id": self._sample_id(sample),
            "repo": getattr(sample, "repo", ""),
            "commit": getattr(sample, "commit", ""),
            "file_path": getattr(sample, "file_path", ""),
            "label": getattr(sample, "label", None),

            "seed_lines": list(
                getattr(sample, "seed_lines", [])
            ),

            "seed_nodes": list(
                getattr(sample, "seed_nodes", [])
            ),

            "function_nodes": list(
                getattr(sample, "function_nodes", [])
            ),

            "cfg": {
                "nodes": self._serialize_nodes(cfg),
                "edges": self._serialize_edges(cfg)
            },

            "pruned_cfg": {
                "nodes": self._serialize_nodes(pruned_cfg),
                "edges": self._serialize_edges(pruned_cfg)
            }
        }

    @staticmethod
    def _serialize_nodes(cfg):
        result = []

        for node in cfg.get("nodes", []):
            result.append({
                "node_id": node.node_id,
                "lineno": getattr(node, "lineno", None),
                "node_type": getattr(node, "node_type", None),
                "text": getattr(node, "text", "")
            })

        return result

    @staticmethod
    def _serialize_edges(cfg):
        return [
            [src, dst]
            for src, dst in cfg.get("edges", [])
        ]

    @staticmethod
    def _sample_id(sample):
        return "{}:{}:{}".format(
            getattr(sample, "repo", ""),
            getattr(sample, "commit", ""),
            getattr(sample, "file_path", "")
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