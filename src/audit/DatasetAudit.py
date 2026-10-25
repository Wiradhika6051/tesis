
import csv
import json
from collections import Counter
from pathlib import Path


class DatasetAudit:
    def __init__(self):
        self.events = []
        self.counts = Counter()

    def record(
        self,
        stage,
        reason,
        sample=None,
        pair_id=None,
        details=None,
    ):
        event = {
            "stage": stage,
            "reason": reason,
            "pair_id": (
                pair_id
                or getattr(sample, "function_pair_id", None)
            ),
            "sample_id": getattr(sample, "sample_id", None),
            "repo": getattr(sample, "repo", None),
            "fix_commit_hash": getattr(
                sample, "fix_commit_hash", None
            ),
            "commit_hash": getattr(sample, "commit_hash", None),
            "file_path": getattr(sample, "file_path", None),
            "revision": getattr(sample, "revision", None),
            "label": getattr(sample, "label", None),
            "details": details or {},
        }

        self.events.append(event)
        self.counts[(stage, reason)] += 1

    def save_events(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = output_dir / "exclusion_events.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as file:
            for event in self.events:
                file.write(json.dumps(event) + "\n")

        csv_path = output_dir / "exclusion_events.csv"
        fields = [
            "stage",
            "reason",
            "pair_id",
            "sample_id",
            "repo",
            "fix_commit_hash",
            "commit_hash",
            "file_path",
            "revision",
            "label",
            "details",
        ]

        with csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()

            for event in self.events:
                row = dict(event)
                row["details"] = json.dumps(row["details"])
                writer.writerow(row)

    def save_summary(self, output_dir, metrics=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "metrics": metrics or {},
            "exclusions": [
                {
                    "stage": stage,
                    "reason": reason,
                    "count": count,
                }
                for (stage, reason), count
                in sorted(self.counts.items())
            ],
        }

        with (output_dir / "dataset_summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(summary, file, indent=2)

    def print_exclusion_summary(self):
        print("\n=== Dataset Exclusion Summary ===")

        if not self.counts:
            print("No exclusions recorded.")
            return

        for (stage, reason), count in sorted(self.counts.items()):
            print(f"{stage} | {reason}: {count}")
