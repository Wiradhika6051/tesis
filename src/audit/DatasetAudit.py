
# File: src/audit/DatasetAudit.py
# Action: CREATE

import csv
import json
import os
from collections import Counter


class DatasetAudit:
    """Collect reproducible dataset-stage statistics and exclusion events."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.report = {
            "counts": {},
            "stages": {},
            "events": [],
        }

    @staticmethod
    def _sample_metadata(sample):
        if sample is None:
            return {}

        return {
            "sample_id": getattr(sample, "sample_id", None),
            "pair_id": getattr(sample, "function_pair_id", None),
            "parent_sample_id": getattr(sample, "parent_sample_id", None),
            "repo": getattr(sample, "repo", None),
            "fix_commit_hash": getattr(sample, "fix_commit_hash", None),
            "commit_hash": getattr(sample, "commit_hash", None),
            "file_path": getattr(sample, "file_path", None),
            "revision": getattr(sample, "revision", None),
            "label": getattr(sample, "label", None),
            "function_name": getattr(sample, "function_name", None),
            "qualified_name": getattr(sample, "qualified_name", None),
        }

    @staticmethod
    def _distinct_pair_count(samples):
        return len(set(
            getattr(sample, "function_pair_id", None)
            for sample in samples
            if getattr(sample, "function_pair_id", None) is not None
        ))

    @staticmethod
    def _missing_pair_id_count(samples):
        return sum(
            1 for sample in samples
            if getattr(sample, "function_pair_id", None) is None
        )

    def record_stage(self, stage_name, samples):
        """Record counts for the exact collection supplied to this method."""
        samples = list(samples)

        labels = Counter(
            getattr(sample, "label", None) for sample in samples
        )

        self.report["stages"][stage_name] = {
            "sample_count": len(samples),
            "pair_count": self._distinct_pair_count(samples),
            "missing_pair_id_count": self._missing_pair_id_count(samples),
            "label_counts": {
                str(label): count for label, count in labels.items()
            },
        }

    def record_event(
        self,
        stage,
        reason,
        sample=None,
        pair_id=None,
        details=None,
        event_type="sample_failure",
    ):
        """Record a failure or exclusion without conflating the two."""
        metadata = self._sample_metadata(sample)

        resolved_pair_id = pair_id
        if resolved_pair_id is None:
            resolved_pair_id = metadata.get("pair_id")

        event = {
            "event_type": event_type,
            "stage": stage,
            "reason": reason,
            "pair_id": resolved_pair_id,
            "details": details or {},
        }
        event.update(metadata)

        self.report["events"].append(event)

        # Event counts are diagnostic counters, not unique excluded samples.
        key = "{}:{}".format(stage, reason)
        self.report["counts"][key] = (
            self.report["counts"].get(key, 0) + 1
        )

        return event

    def record_pair_exclusion(
        self, stage, pair_id, members, reason="incomplete_pair"
    ):
        """Record one pair exclusion and describe the members remaining."""
        members = list(members)

        details = {
            "member_count": len(members),
            "sample_ids": [
                getattr(sample, "sample_id", None)
                for sample in members
            ],
            "revisions": [
                getattr(sample, "revision", None)
                for sample in members
            ],
            "labels": [
                getattr(sample, "label", None)
                for sample in members
            ],
            "qualified_names": [
                getattr(sample, "qualified_name", None)
                for sample in members
            ],
        }

        event = self.record_event(
            stage=stage,
            reason=reason,
            pair_id=pair_id,
            details=details,
            event_type="pair_exclusion",
        )

        key = "{}:dropped_pairs".format(stage)
        self.report["counts"][key] = (
            self.report["counts"].get(key, 0) + 1
        )

        return event

    def record_missing_pair_id(self, stage, sample):
        return self.record_event(
            stage=stage,
            reason="missing_pair_id",
            sample=sample,
            details={
                "excluded_from_pair_accounting": True,
            },
            event_type="sample_exclusion",
        )

    def snapshot(self):
        """Return an independent snapshot safe to retain per dataset split."""
        return json.loads(json.dumps(self.report))

    def export_json(self, path):
        self._ensure_parent(path)
        with open(path, "w") as handle:
            json.dump(self.report, handle, indent=2, sort_keys=True)

    def export_jsonl(self, path):
        self._ensure_parent(path)
        with open(path, "w") as handle:
            for event in self.report["events"]:
                handle.write(json.dumps(event, sort_keys=True) + "\n")

    def export_csv(self, path):
        self._ensure_parent(path)

        events = self.report["events"]
        columns = [
            "event_type",
            "stage",
            "reason",
            "sample_id",
            "pair_id",
            "parent_sample_id",
            "repo",
            "fix_commit_hash",
            "commit_hash",
            "file_path",
            "revision",
            "label",
            "function_name",
            "qualified_name",
            "details",
        ]

        with open(path, "w") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=columns, extrasaction="ignore"
            )
            writer.writeheader()

            for event in events:
                row = dict(event)
                row["details"] = json.dumps(
                    row.get("details", {}), sort_keys=True
                )
                writer.writerow(row)

    @staticmethod
    def read_json(path):
        with open(path, "r") as handle:
            return json.load(handle)

    @staticmethod
    def read_jsonl(path):
        events = []
        with open(path, "r") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except ValueError as exc:
                    raise ValueError(
                        "Invalid JSONL at line {}: {}".format(
                            line_number, exc
                        )
                    )
        return events

    @staticmethod
    def read_csv(path):
        with open(path, "r") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _ensure_parent(path):
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent)
