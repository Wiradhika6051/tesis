import csv
import json
import os


class PruningDebugExporter:

    def __init__(self, output_dir):
        self.output_dir = output_dir

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    def export(self, samples, pruner_name):

        output_path = os.path.join(
            self.output_dir,
            "{}.csv".format(pruner_name)
        )

        rows = [
            self._build_row(sample)
            for sample in samples
            if sample.cfg
            and sample.pruned_cfg
        ]

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

        nodes_removed = self._get_removed_nodes(
            nodes_before,
            nodes_after
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
                self._get_node_text(cfg)
            ),

            "text_after": self._serialize(
                self._get_node_text(pruned_cfg)
            ),

            "node_count_before": len(
                nodes_before
            ),

            "node_count_after": len(
                nodes_after
            )
        }

    @staticmethod
    def _get_sample_id(sample):

        return "{}:{}:{}".format(
            getattr(
                sample,
                "repo",
                ""
            ),
            getattr(
                sample,
                "commit",
                ""
            ),
            getattr(
                sample,
                "file_path",
                ""
            )
        )

    @staticmethod
    def _get_node_ids(cfg):

        return [
            node.node_id
            for node in cfg.get(
                "nodes",
                []
            )
        ]

    @staticmethod
    def _get_node_text(cfg):

        return [
            "{}: {}".format(
                node.node_id,
                node.text
            )
            for node in cfg.get(
                "nodes",
                []
            )
        ]

    @staticmethod
    def _get_removed_nodes(
        nodes_before,
        nodes_after
    ):

        nodes_after = set(
            nodes_after
        )

        return [
            node_id
            for node_id in nodes_before
            if node_id not in nodes_after
        ]

    @staticmethod
    def _serialize(value):

        return json.dumps(
            value,
            ensure_ascii=False
        )

    @staticmethod
    def _write_csv(
        path,
        rows
    ):

        fieldnames = [
            "sample_id",
            "repo",
            "file_path",
            "commit",
            "label",
            "seed_lines",
            "seed_nodes",
            "nodes_before",
            "nodes_after",
            "nodes_removed",
            "text_before",
            "text_after",
            "node_count_before",
            "node_count_after"
        ]

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