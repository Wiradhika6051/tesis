
import csv
from pathlib import Path


class MatchAudit:
    FIELDS = [
        "repo",
        "fix_commit_hash",
        "file_path",
        "vulnerable_function",
        "fixed_function",
        "match_method",
        "similarity",
        "runner_up_similarity",
        "status",
        "reason",
    ]

    def __init__(self):
        self.events = []

    def record(
        self,
        old_sample=None,
        new_sample=None,
        match_method=None,
        similarity=None,
        runner_up_similarity=None,
        status="rejected",
        reason=None,
    ):
        self.events.append({
            "repo": getattr(
                old_sample or new_sample, "repo", None
            ),
            "fix_commit_hash": getattr(
                old_sample or new_sample, "fix_commit_hash", None
            ),
            "file_path": getattr(
                old_sample or new_sample, "file_path", None
            ),
            "vulnerable_function": getattr(
                old_sample, "qualified_name", None
            ),
            "fixed_function": getattr(
                new_sample, "qualified_name", None
            ),
            "match_method": match_method,
            "similarity": similarity,
            "runner_up_similarity": runner_up_similarity,
            "status": status,
            "reason": reason,
        })

    def save_csv(self, output_path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=self.FIELDS
            )
            writer.writeheader()
            writer.writerows(self.events)
