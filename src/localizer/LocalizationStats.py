from collections import Counter


def summarize_localization(samples):
    """
    Summarize localization outcomes across a collection of samples.

    Samples must have been processed by LineCFGLocalizer.
    """
    counts = Counter()
    total_unmatched_lines = 0

    for sample in samples:
        status = sample.localization_status

        counts[status] += 1
        total_unmatched_lines += len(
            sample.unmatched_seed_lines
        )

    total = len(samples)
    failed = counts["failed"]
    partial = counts["partial"]
    no_changed_lines = counts["no_changed_lines"]

    return {
        "total_samples": total,
        "successful_samples": counts["success"],
        "partially_localized_samples": partial,
        "failed_samples": failed,
        "no_changed_lines_samples": no_changed_lines,
        "excluded_samples": failed + no_changed_lines,
        "unmatched_changed_lines": total_unmatched_lines,
        "failure_rate": (
            failed / float(total) if total else 0.0
        ),
        "partial_localization_rate": (
            partial / float(total) if total else 0.0
        ),
    }