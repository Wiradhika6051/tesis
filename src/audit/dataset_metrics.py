
def unique_fixing_commits(samples):
    return {
        sample.fix_commit_hash
        for sample in samples
        if getattr(sample, "fix_commit_hash", None)
    }


def count_function_pairs(samples):
    return len({
        sample.function_pair_id
        for sample in samples
        if getattr(sample, "function_pair_id", None) is not None
    })


def validate_final_pairs(samples):
    """
    Raise an error if the final dataset contains incomplete pairs.
    """
    from collections import defaultdict

    grouped = defaultdict(list)

    for sample in samples:
        pair_id = getattr(sample, "function_pair_id", None)

        if pair_id is None:
            raise ValueError(
                "Final dataset contains a sample without function_pair_id"
            )

        grouped[pair_id].append(sample)

    for pair_id, members in grouped.items():
        labels = sorted(sample.label for sample in members)

        if len(members) != 2 or labels != [0, 1]:
            raise ValueError(
                f"Incomplete pair {pair_id}: "
                f"{len(members)} members, labels={labels}"
            )


def build_dataset_metrics(
    original_file_samples,
    function_candidates,
    matched_samples,
    eligible_samples,
    final_samples,
):
    validate_final_pairs(final_samples)

    return {
        "original_fixing_commits": len(
            unique_fixing_commits(original_file_samples)
        ),
        "original_file_revision_samples": len(
            original_file_samples
        ),
        "extracted_function_candidates": len(
            function_candidates
        ),
        "matched_function_pairs": count_function_pairs(
            matched_samples
        ),
        "eligible_function_pairs": count_function_pairs(
            eligible_samples
        ),
        "final_function_samples": len(final_samples),
        "final_function_pairs": count_function_pairs(
            final_samples
        ),
    }
