
from collections import defaultdict


def retain_complete_pairs(samples, audit, stage):
    """
    Keep only pairs containing exactly two members with labels 0 and 1.

    Returns:
        retained_samples, dropped_samples
    """
    grouped = defaultdict(list)
    dropped = []

    for sample in samples:
        pair_id = getattr(sample, "function_pair_id", None)

        if pair_id is None:
            audit.record(
                stage=stage,
                reason="missing_pair_id",
                sample=sample,
            )
            dropped.append(sample)
            continue

        grouped[pair_id].append(sample)

    retained = []

    for pair_id, members in grouped.items():
        labels = [sample.label for sample in members]

        is_complete = (
            len(members) == 2
            and sorted(labels) == [0, 1]
        )

        if is_complete:
            retained.extend(members)
            continue

        audit.record(
            stage=stage,
            reason="incomplete_pair",
            pair_id=pair_id,
            details={
                "member_count": len(members),
                "labels": labels,
                "sample_ids": [
                    getattr(member, "sample_id", None)
                    for member in members
                ],
            },
        )
        dropped.extend(members)

    return retained, dropped
