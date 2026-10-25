from types import SimpleNamespace
import json

import pytest

from src.pipeline.function_dataset_preparation_pipeline import (
    FunctionDatasetPreparationPipeline,
)


# These tests exercise the real pipeline orchestration with deterministic
# collaborators. They do not need a real repository, AST parser, or CFG model.


def make_sample(sample_id, pair_id, revision, label):
    return SimpleNamespace(
        sample_id=sample_id,
        function_pair_id=pair_id,
        parent_sample_id=None,
        repo="phase3-test-repo",
        fix_commit_hash="fix-" + str(pair_id),
        commit_hash="commit-" + revision + "-" + sample_id,
        file_path="example.py",
        revision=revision,
        label=label,
        function_name="target",
        qualified_name="example.target",
        source="def target():\n    return 1\n",
        function_source="def target():\n    return 1\n",
        seed_lines=[2],
        seed_nodes=[],
        function_nodes=[],
        cfg=None,
        pruned_cfg=None,
    )


def make_pair(pair_id):
    return [
        make_sample(pair_id + "-v", pair_id, "vulnerable", 1),
        make_sample(pair_id + "-f", pair_id, "fixed", 0),
    ]


class FakeDiffLocalizer:
    def localize(self, sample):
        return getattr(sample, "seed_lines", [])


class FakeFunctionSampleBuilder:
    def build(self, sample):
        # This test focuses on pipeline audit behavior; one function candidate
        # per input sample keeps expected counts easy to verify.
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
        return {
            "nodes": [
                SimpleNamespace(node_id=0, lineno=1, end_lineno=2)
            ],
            "edges": [],
        }


class FakeCFGLocalizer:
    def __init__(self, no_seed_sample_ids=None):
        self.no_seed_sample_ids = set(no_seed_sample_ids or [])

    def localize(self, sample):
        sample.seed_nodes = (
            [] if sample.sample_id in self.no_seed_sample_ids else [0]
        )
        return sample


class FakeFunctionScopeLocalizer:
    def __init__(self, no_scope_sample_ids=None):
        self.no_scope_sample_ids = set(no_scope_sample_ids or [])

    def localize(self, sample):
        sample.function_nodes = (
            [] if sample.sample_id in self.no_scope_sample_ids else [0]
        )
        return sample


class FakePruner:
    def __init__(self, empty_pruned_sample_ids=None):
        self.empty_pruned_sample_ids = set(empty_pruned_sample_ids or [])

    def prune(self, sample):
        nodes = (
            []
            if sample.sample_id in self.empty_pruned_sample_ids
            else [{"id": 0}]
        )
        sample.pruned_cfg = {"nodes": nodes, "edges": []}
        return sample


def make_pipeline(
    monkeypatch,
    cfg_fail_ids=None,
    no_seed_ids=None,
    no_scope_ids=None,
    empty_pruned_ids=None,
):
    # Fake samples do not contain every field expected by the project's
    # console statistics printer. Disable only that display helper; the real
    # pipeline orchestration and audit logic still run.
    monkeypatch.setattr(
        "src.pipeline.function_dataset_preparation_pipeline.print_statistics",
        lambda *args, **kwargs: None,
    )

    return FunctionDatasetPreparationPipeline(
        function_sample_builder=FakeFunctionSampleBuilder(),
        paired_function_extractor=FakePairExtractor(),
        cfg_builder=FakeCFGBuilder(cfg_fail_ids),
        diff_localizer=FakeDiffLocalizer(),
        cfg_localizer=FakeCFGLocalizer(no_seed_ids),
        function_scope_localizer=FakeFunctionScopeLocalizer(no_scope_ids),
        pruner=FakePruner(empty_pruned_ids),
    )


def event_matches(audit, reason, stage=None, event_type=None):
    return [
        event for event in audit["events"]
        if event.get("reason") == reason
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


def test_end_to_end_success_counts_match_actual_collections(monkeypatch):
    pipeline = make_pipeline(monkeypatch)
    raw_samples = make_pair("pair-ok")

    prepared = pipeline.prepare(raw_samples)
    audit = pipeline.audit_snapshot()

    assert_complete_pairs(prepared, {"pair-ok"})
    assert len(prepared) == 2

    # These are the exact collections represented by these stage snapshots.
    assert audit["stages"]["raw"]["sample_count"] == len(raw_samples)
    assert audit["stages"]["raw"]["pair_count"] == 1
    assert audit["stages"]["pairing_before_pair_validation"]["sample_count"] == 2
    assert audit["stages"]["pairing"]["sample_count"] == len(prepared)
    assert audit["stages"]["pruning"]["sample_count"] == len(prepared)
    assert audit["stages"]["pruning"]["pair_count"] == 1
    assert audit["stages"]["raw"]["missing_pair_id_count"] == 0

    assert not event_matches(audit, "cfg_build_failed")
    assert not event_matches(audit, "incomplete_pair", event_type="pair_exclusion")


def test_cfg_failure_records_sample_failure_and_one_pair_exclusion(monkeypatch):
    samples = make_pair("pair-cfg-fail")
    failed_sample_id = "pair-cfg-fail-v"
    pipeline = make_pipeline(monkeypatch, cfg_fail_ids={failed_sample_id})

    prepared = pipeline.prepare(samples)
    audit = pipeline.audit_snapshot()

    assert prepared == []

    failures = event_matches(
        audit, "cfg_build_failed", stage="cfg_construction"
    )
    assert len(failures) == 1
    assert failures[0]["sample_id"] == failed_sample_id
    assert failures[0]["pair_id"] == "pair-cfg-fail"
    assert failures[0]["event_type"] == "sample_failure"

    pair_exclusions = event_matches(
        audit,
        "incomplete_pair",
        stage="cfg_construction",
        event_type="pair_exclusion",
    )
    assert len(pair_exclusions) == 1
    assert pair_exclusions[0]["pair_id"] == "pair-cfg-fail"
    assert pair_exclusions[0]["details"]["member_count"] == 1
    assert pair_exclusions[0]["details"]["sample_ids"] == [
        "pair-cfg-fail-f"
    ]
    assert audit["counts"]["cfg_construction_pairs_dropped"] == 1


def test_missing_pair_id_is_separate_from_dropped_pairs(monkeypatch):
    samples = make_pair("pair-valid")
    orphan = make_sample("orphan", None, "vulnerable", 1)
    samples.append(orphan)
    pipeline = make_pipeline(monkeypatch)

    prepared = pipeline.prepare(samples)
    audit = pipeline.audit_snapshot()

    assert_complete_pairs(prepared, {"pair-valid"})
    pairing_input = audit["stages"]["pairing_before_pair_validation"]
    assert pairing_input["sample_count"] == 3
    assert pairing_input["pair_count"] == 1
    assert pairing_input["missing_pair_id_count"] == 1

    missing_id_events = event_matches(
        audit, "missing_pair_id", stage="pairing",
        event_type="sample_exclusion",
    )
    assert len(missing_id_events) == 1
    assert missing_id_events[0]["sample_id"] == "orphan"
    assert audit["counts"]["pairing_pairs_dropped"] == 0


def test_localization_failure_excludes_pair_and_is_audited(monkeypatch):
    samples = make_pair("pair-no-seeds")
    pipeline = make_pipeline(
        monkeypatch, no_seed_ids={"pair-no-seeds-f"}
    )

    prepared = pipeline.prepare(samples)
    audit = pipeline.audit_snapshot()

    assert prepared == []
    failures = event_matches(
        audit, "no_seed_nodes", stage="cfg_localization"
    )
    assert len(failures) == 1
    assert failures[0]["sample_id"] == "pair-no-seeds-f"

    pair_exclusions = event_matches(
        audit, "incomplete_pair", stage="cfg_localization",
        event_type="pair_exclusion",
    )
    assert len(pair_exclusions) == 1
    assert pair_exclusions[0]["pair_id"] == "pair-no-seeds"


def test_pruning_failure_excludes_pair_and_is_audited(monkeypatch):
    samples = make_pair("pair-prune-fail")
    pipeline = make_pipeline(
        monkeypatch, empty_pruned_ids={"pair-prune-fail-v"}
    )

    prepared = pipeline.prepare(samples)
    audit = pipeline.audit_snapshot()

    assert prepared == []
    failures = event_matches(
        audit, "pruned_cfg_has_no_nodes", stage="pruning"
    )
    assert len(failures) == 1
    assert failures[0]["sample_id"] == "pair-prune-fail-v"

    pair_exclusions = event_matches(
        audit, "incomplete_pair", stage="pruning",
        event_type="pair_exclusion",
    )
    assert len(pair_exclusions) == 1
    assert pair_exclusions[0]["pair_id"] == "pair-prune-fail"


def test_train_validation_test_audit_snapshots_are_independent(monkeypatch):
    pipeline = make_pipeline(monkeypatch)

    train_samples = make_pair("train-pair")
    train_result = pipeline.prepare(train_samples)
    train_audit = pipeline.audit_snapshot()

    val_samples = make_pair("val-pair")
    val_result = pipeline.prepare(val_samples)
    val_audit = pipeline.audit_snapshot()

    test_samples = make_pair("test-pair")
    test_result = pipeline.prepare(test_samples)
    test_audit = pipeline.audit_snapshot()

    # Mutating the live audit after the test split must not affect snapshots.
    pipeline.audit["events"].append({"reason": "later-mutation"})
    pipeline.audit["stages"].clear()

    assert_complete_pairs(train_result, {"train-pair"})
    assert_complete_pairs(val_result, {"val-pair"})
    assert_complete_pairs(test_result, {"test-pair"})

    assert train_audit["stages"]["raw"]["sample_count"] == 2
    assert val_audit["stages"]["raw"]["sample_count"] == 2
    assert test_audit["stages"]["raw"]["sample_count"] == 2
    assert train_audit["events"] == []
    assert val_audit["events"] == []
    assert test_audit["events"] == []
    assert "raw" in train_audit["stages"]
    assert "raw" in val_audit["stages"]
    assert "raw" in test_audit["stages"]


def test_audit_snapshot_can_be_serialized_and_reloaded(tmp_path, monkeypatch):
    pipeline = make_pipeline(monkeypatch, cfg_fail_ids={"pair-json-v"})
    pipeline.prepare(make_pair("pair-json"))
    snapshot = pipeline.audit_snapshot()

    output_path = tmp_path / "pipeline_audit.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(snapshot, file, indent=2)

    with output_path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)

    assert loaded == snapshot
    assert loaded["stages"]["raw"]["sample_count"] == 2
    assert len(event_matches(loaded, "cfg_build_failed")) == 1
