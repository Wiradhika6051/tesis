"""Integration tests for CFG localization and pruning in the dataset pipeline.

These tests exercise the real pipeline orchestration with deterministic fake
collaborators. They do not require a repository, parser, or real CFG builder.

Run from the repository root:
    python -m pytest src/tests/test_cfg_pruning_pipeline_integration.py -v
"""

from types import SimpleNamespace

import pytest

from src.pipeline.function_dataset_preparation_pipeline import (
    FunctionDatasetPreparationPipeline
)


def make_sample(sample_id, pair_id, revision, label, seed_lines=None):
    return SimpleNamespace(
        sample_id=sample_id,
        function_pair_id=pair_id,
        parent_sample_id=None,
        repo="cfg-pipeline-test-repo",
        fix_commit_hash="fix-" + str(pair_id),
        commit_hash="commit-" + revision + "-" + sample_id,
        file_path="example.py",
        revision=revision,
        label=label,
        function_name="target",
        qualified_name="example.target",
        source="def target():\n    return 1\n",
        function_source="def target():\n    return 1\n",
        seed_lines=[2] if seed_lines is None else list(seed_lines),
        seed_nodes=[],
        function_nodes=[],
        cfg=None,
        pruned_cfg=None,
    )


def make_pair(pair_id, vulnerable_seed_lines=None, fixed_seed_lines=None):
    return [
        make_sample(
            pair_id + "-v", pair_id, "vulnerable", 1,
            vulnerable_seed_lines,
        ),
        make_sample(
            pair_id + "-f", pair_id, "fixed", 0,
            fixed_seed_lines,
        ),
    ]


class FakeDiffLocalizer:
    def localize(self, sample):
        return list(getattr(sample, "seed_lines", []) or [])


class FakeFunctionSampleBuilder:
    def build(self, sample):
        return [sample]


class FakePairExtractor:
    def __init__(self):
        self.audit = {"counts": {}}

    def extract(self, samples):
        return list(samples)


class FakeCFGBuilder:
    def __init__(self, fail_sample_ids=None):
        self.fail_sample_ids = set(fail_sample_ids or [])

    def build(self, sample):
        if sample.sample_id in self.fail_sample_ids:
            return None

        # Node 0 contains source line 2. Node 1 is a separate AST node.
        return {
            "nodes": [
                SimpleNamespace(
                    node_id=0, lineno=1, end_lineno=2,
                    node_type="FunctionDef", text="def target(): ...",
                ),
                SimpleNamespace(
                    node_id=1, lineno=2, end_lineno=2,
                    node_type="Return", text="return 1",
                ),
                SimpleNamespace(
                    node_id=2, lineno=-1, end_lineno=-1,
                    node_type="ENTRY", text="",
                ),
            ],
            "edges": [(2, 0), (0, 1)],
        }


class FakeCFGLocalizer:
    """Set realistic localization metadata for pipeline integration tests."""

    def __init__(self, outcomes=None):
        # sample_id -> (status, seed_nodes, unmatched_lines, exclusion_reason)
        self.outcomes = outcomes or {}

    def localize(self, sample):
        outcome = self.outcomes.get(
            sample.sample_id,
            ("success", [0, 1], [], None),
        )
        status, seed_nodes, unmatched_lines, exclusion_reason = outcome

        sample.seed_nodes = list(seed_nodes)
        sample.unmatched_seed_lines = list(unmatched_lines)
        sample.localization_status = status
        sample.exclusion_reason = exclusion_reason
        sample.line_to_node = {
            line: ([0] if line != 999 else [])
            for line in (sample.seed_lines or [])
        }
        return sample


class FakeFunctionScopeLocalizer:
    def __init__(self, no_scope_sample_ids=None):
        self.no_scope_sample_ids = set(no_scope_sample_ids or [])

    def localize(self, sample):
        sample.function_nodes = (
            [] if sample.sample_id in self.no_scope_sample_ids
            else [0, 1]
        )
        return sample


class FakePruner:
    def __init__(self, outcomes=None):
        # sample_id -> None, empty nodes, or explicit pruned CFG
        self.outcomes = outcomes or {}

    def prune(self, sample):
        outcome = self.outcomes.get(sample.sample_id, "success")

        if outcome == "none":
            sample.pruned_cfg = None
        elif outcome == "empty":
            sample.pruned_cfg = {"nodes": [], "edges": []}
        elif isinstance(outcome, dict):
            sample.pruned_cfg = outcome
        else:
            sample.pruned_cfg = {
                "nodes": [
                    node for node in sample.cfg["nodes"]
                    if node.node_id in set(sample.seed_nodes)
                ],
                "edges": [],
            }
        return sample


def make_pipeline(
    monkeypatch,
    cfg_fail_ids=None,
    localization_outcomes=None,
    no_scope_ids=None,
    pruning_outcomes=None,
):
    # Silence only the display helper; the actual pipeline and audit run.
    monkeypatch.setattr(
        "src.pipeline.function_dataset_preparation_pipeline.print_statistics",
        lambda *args, **kwargs: None,
    )

    return FunctionDatasetPreparationPipeline(
        function_sample_builder=FakeFunctionSampleBuilder(),
        paired_function_extractor=FakePairExtractor(),
        cfg_builder=FakeCFGBuilder(cfg_fail_ids),
        diff_localizer=FakeDiffLocalizer(),
        cfg_localizer=FakeCFGLocalizer(localization_outcomes),
        function_scope_localizer=FakeFunctionScopeLocalizer(no_scope_ids),
        pruner=FakePruner(pruning_outcomes),
    )


def events(audit, reason=None, stage=None, event_type=None):
    return [
        event for event in audit["events"]
        if (reason is None or event.get("reason") == reason)
        and (stage is None or event.get("stage") == stage)
        and (event_type is None or event.get("event_type") == event_type)
    ]


def assert_complete_pairs(samples, expected_pair_ids):
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample.function_pair_id, []).append(sample)

    assert set(grouped) == set(expected_pair_ids)
    for pair_id, members in grouped.items():
        assert len(members) == 2, pair_id
        assert {member.label for member in members} == {0, 1}, pair_id
        assert {member.revision for member in members} == {
            "vulnerable", "fixed"
        }, pair_id


def test_successful_multi_seed_samples_reach_pruner_and_remain_paired(monkeypatch):
    pipeline = make_pipeline(monkeypatch)
    prepared = pipeline.prepare(make_pair("multi-seed"))
    audit = pipeline.audit_snapshot()

    assert_complete_pairs(prepared, {"multi-seed"})
    assert len(prepared) == 2
    assert all(sample.seed_nodes == [0, 1] for sample in prepared)
    assert all(sample.pruned_cfg["nodes"] for sample in prepared)
    assert audit["stages"]["cfg_localization_outcomes"] == {
        "sample_count": 2,
        "success_count": 2,
        "partial_count": 0,
        "failed_count": 0,
        "no_changed_lines_count": 0,
        "unknown_status_count": 0,
        "excluded_count": 0,
        "unmatched_changed_line_count": 0,
        "failure_rate": 0.0,
        "partial_rate": 0.0,
        "no_changed_lines_rate": 0.0,
    }


def test_partial_localization_is_recorded_but_sample_can_continue(monkeypatch):
    outcomes = {
        "partial-v": ("partial", [0], [999], None),
    }
    pipeline = make_pipeline(
        monkeypatch,
        localization_outcomes=outcomes,
    )

    prepared = pipeline.prepare(make_pair("partial"))
    audit = pipeline.audit_snapshot()

    # The other pair member localizes successfully, so the pair remains valid.
    assert_complete_pairs(prepared, {"partial"})
    warning = events(
        audit,
        reason="unmatched_changed_lines",
        stage="cfg_localization",
        event_type="localization_warning",
    )
    assert len(warning) == 1
    assert warning[0]["sample_id"] == "partial-v"
    assert warning[0]["details"]["unmatched_seed_lines"] == [999]

    summary = audit["stages"]["cfg_localization_outcomes"]
    assert summary["sample_count"] == 2
    assert summary["partial_count"] == 1
    assert summary["success_count"] == 1
    assert summary["unmatched_changed_line_count"] == 1
    assert summary["partial_rate"] == 0.5


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("failed", "no_valid_seed_nodes"),
        ("no_changed_lines", "no_changed_lines"),
    ],
)
def test_total_localization_failure_is_recorded_and_pair_excluded(
    monkeypatch, status, reason
):
    outcomes = {
        "bad-v": (
            status,
            [],
            [2] if status == "failed" else [],
            "no_valid_seed_nodes" if status == "failed" else "no_changed_lines",
        ),
    }
    pipeline = make_pipeline(
        monkeypatch,
        localization_outcomes=outcomes,
    )

    prepared = pipeline.prepare(make_pair("bad"))
    audit = pipeline.audit_snapshot()

    assert prepared == []
    assert len(events(
        audit,
        reason=reason,
        stage="cfg_localization",
        event_type="sample_exclusion",
    )) == 1

    # The outcome summary includes both the excluded and retained-at-localizer
    # samples, not merely the samples that survive localization.
    summary = audit["stages"]["cfg_localization_outcomes"]
    assert summary["sample_count"] == 2
    assert summary["excluded_count"] == 1
    if status == "failed":
        assert summary["failed_count"] == 1
        assert summary["failure_rate"] == 0.5
    else:
        assert summary["no_changed_lines_count"] == 1
        assert summary["no_changed_lines_rate"] == 0.5

    assert len(events(
        audit,
        reason="incomplete_pair",
        stage="cfg_localization",
        event_type="pair_exclusion",
    )) == 1


@pytest.mark.parametrize(
    ("pruner_outcome", "reason"),
    [
        ("none", "pruned_cfg_is_none"),
        ("empty", "pruned_cfg_has_no_nodes"),
    ],
)
def test_invalid_pruned_cfg_is_recorded_and_pair_excluded(
    monkeypatch, pruner_outcome, reason
):
    pipeline = make_pipeline(
        monkeypatch,
        pruning_outcomes={"prune-v": pruner_outcome},
    )

    prepared = pipeline.prepare(make_pair("prune"))
    audit = pipeline.audit_snapshot()

    assert prepared == []
    failures = events(
        audit,
        reason=reason,
        stage="pruning",
    )
    assert len(failures) == 1
    assert failures[0]["sample_id"] == "prune-v"

    pair_exclusions = events(
        audit,
        reason="incomplete_pair",
        stage="pruning",
        event_type="pair_exclusion",
    )
    assert len(pair_exclusions) == 1
    assert pair_exclusions[0]["pair_id"] == "prune"


def test_function_scope_failure_is_distinct_from_cfg_localization_failure(
    monkeypatch,
):
    pipeline = make_pipeline(
        monkeypatch,
        no_scope_ids={"scope-v"},
    )

    prepared = pipeline.prepare(make_pair("scope"))
    audit = pipeline.audit_snapshot()

    assert prepared == []
    failures = events(
        audit,
        reason="no_valid_function_seed_nodes",
        stage="function_scope",
    )
    assert len(failures) == 1
    assert failures[0]["sample_id"] == "scope-v"
    assert not events(audit, stage="cfg_localization", reason="no_valid_seed_nodes")


def test_pair_survives_when_both_samples_prune_successfully(monkeypatch):
    pipeline = make_pipeline(monkeypatch)
    prepared = pipeline.prepare(make_pair("pair-ok"))
    audit = pipeline.audit_snapshot()

    assert_complete_pairs(prepared, {"pair-ok"})
    assert audit["stages"]["pruning_before_pair_validation"]["sample_count"] == 2
    assert audit["stages"]["pruning"]["sample_count"] == 2
    assert not events(audit, stage="pruning", reason="pruned_cfg_is_none")
    assert not events(audit, stage="pruning", reason="pruned_cfg_has_no_nodes")
