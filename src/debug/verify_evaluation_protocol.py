
from collections import defaultdict


def verify_group_split(records, group_key, split_key="split"):
    """
    records: iterable of dictionaries containing a stable group ID
             and a split assignment.
    group_key: field identifying the originating change/commit group.
    split_key: field containing train/validation/test.
    """
    group_splits = defaultdict(set)
    missing_groups = []

    for index, record in enumerate(records):
        group_id = record.get(group_key)
        split = record.get(split_key)

        if group_id is None or group_id == "":
            missing_groups.append(index)
            continue

        if split not in {"train", "validation", "test"}:
            raise ValueError(
                f"Invalid split at record {index}: {split!r}"
            )

        group_splits[group_id].add(split)

    leaked_groups = {
        group_id: sorted(splits)
        for group_id, splits in group_splits.items()
        if len(splits) > 1
    }

    print(f"Records: {len(records)}")
    print(f"Groups: {len(group_splits)}")
    print(f"Records missing group IDs: {len(missing_groups)}")
    print(f"Groups crossing splits: {len(leaked_groups)}")

    if missing_groups:
        print("Missing group IDs at record indices:", missing_groups[:20])

    if leaked_groups:
        for group_id, splits in list(leaked_groups.items())[:20]:
            print(f"LEAKAGE: {group_id!r} appears in {splits}")

    assert not missing_groups, "Some records have no group ID."
    assert not leaked_groups, "Some groups occur in multiple splits."

    print("PASS: every group belongs to exactly one split.")
