from types import SimpleNamespace

from src.localizer.LocalizationStats import (
    summarize_localization,
)


def make_result(status, unmatched_lines=None):
    return SimpleNamespace(
        localization_status=status,
        unmatched_seed_lines=unmatched_lines or [],
    )


def test_localization_statistics():
    samples = [
        make_result("success"),
        make_result("success"),
        make_result("partial", [99]),
        make_result("failed", [10, 20]),
        make_result("no_changed_lines"),
    ]

    stats = summarize_localization(samples)

    assert stats["total_samples"] == 5
    assert stats["successful_samples"] == 2
    assert stats["partially_localized_samples"] == 1
    assert stats["failed_samples"] == 1
    assert stats["no_changed_lines_samples"] == 1
    assert stats["excluded_samples"] == 2
    assert stats["unmatched_changed_lines"] == 3
    assert stats["failure_rate"] == 0.2
    assert stats["partial_localization_rate"] == 0.2


def test_empty_dataset_has_zero_rates():
    stats = summarize_localization([])

    assert stats["total_samples"] == 0
    assert stats["failure_rate"] == 0.0
    assert stats["partial_localization_rate"] == 0.0