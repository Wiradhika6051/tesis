import csv
import json
import os

from src.vocabulary import tokenize_code


class PruningDebugExporter:

    def __init__(self, output_dir):
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def export(self, samples, pruner_name):
        output_path = os.path.join(
            self.output_dir,
            "{}.csv".format(pruner_name)
        )

        rows = []

        for sample in samples:
            rows.append(
                self._build_row(sample)
            )

        if not rows:
            print(
                "[PRUNING DEBUG] No samples to export: {}".format(
                    pruner_name
                )
            )
            return

        self._write_csv(
            output_path,
            rows
        )

        print(
            "[PRUNING DEBUG] Exported {} samples -> {}".format(
                len(rows),
                output_path
            )
        )

    def _build_row(self, sample):

        cfg = sample.cfg
        pruned_cfg = sample.pruned_cfg

        nodes_before = self._get_node_ids(
            cfg
        )

        nodes_after = self._get_node_ids(
            pruned_cfg
        )

        nodes_after_set = set(
            nodes_after
        )

        nodes_removed = [
            node_id
            for node_id in nodes_before
            if node_id not in nodes_after_set
        ]

        text_before = self._get_node_text(
            cfg
        )

        text_after = self._get_node_text(
            pruned_cfg
        )

        tokens_before = self._get_tokens(
            cfg
        )

        tokens_after = self._get_tokens(
            pruned_cfg
        )

        return {
            "sample_id": self._get_sample_id(
                sample
            ),

            "repo": getattr(
                sample,
                "repo",
                ""
            ),

            "file_path": getattr(
                sample,
                "file_path",
                ""
            ),

            "commit": getattr(
                sample,
                "commit",
                ""
            ),

            "label": getattr(
                sample,
                "label",
                ""
            ),

            "source_code": getattr(
                sample,
                "source",
                ""
            ),

            "seed_lines": self._serialize(
                getattr(
                    sample,
                    "seed_lines",
                    []
                )
            ),

            "seed_nodes": self._serialize(
                getattr(
                    sample,
                    "seed_nodes",
                    []
                )
            ),

            "nodes_before": self._serialize(
                nodes_before
            ),

            "nodes_after": self._serialize(
                nodes_after
            ),

            "nodes_removed": self._serialize(
                nodes_removed
            ),

            "text_before": self._serialize(
                text_before
            ),

            "text_after": self._serialize(
                text_after
            ),

            "tokens_before": self._serialize(
                tokens_before
            ),

            "tokens_after": self._serialize(
                tokens_after
            ),

            "node_count_before": len(
                nodes_before
            ),

            "node_count_after": len(
                nodes_after
            ),

            "edge_count_before": len(
                cfg.get("edges", [])
            ),

            "edge_count_after": len(
                pruned_cfg.get("edges", [])
            )
        }

    @staticmethod
    def _get_sample_id(sample):
        return "{}:{}:{}".format(
            getattr(sample, "repo", ""),
            getattr(sample, "commit", ""),
            getattr(sample, "file_path", "")
        )

    @staticmethod
    def _get_node_ids(cfg):
        return [
            node.node_id
            for node in cfg.get("nodes", [])
        ]

    @staticmethod
    def _get_node_text(cfg):
        return [
            node.text
            for node in cfg.get("nodes", [])
        ]

    @staticmethod
    def _get_tokens(cfg):

        code = "\n".join(
            node.text
            for node in cfg.get("nodes", [])
            if node.text and node.text.strip()
        )

        if not code.strip():
            return []

        return tokenize_code(
            code
        )

    @staticmethod
    def _serialize(value):
        return json.dumps(
            value,
            ensure_ascii=False
        )

    @staticmethod
    def _write_csv(path, rows):

        fieldnames = list(
            rows[0].keys()
        )

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(rows)