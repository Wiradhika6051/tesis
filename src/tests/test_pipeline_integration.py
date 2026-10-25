
from types import SimpleNamespace

import pytest

from src.pipeline.function_dataset_preparation_pipeline import (
    FunctionDatasetPreparationPipeline,
)


# ---------------------------------------------------------
# Deterministic fixture components
# ---------------------------------------------------------

def make_sample(sample_id, pair_id, revision, label):
    return SimpleNamespace(
        sample_id=sample_id,
        function_pair_id=pair_id,
        parent_sample_id=None,
        repo="integration-test-repo",
        fix_commit_hash="fix-" + pair_id,
        commit_hash="commit-" + revision + "-" + pair_id,
        file_path="example.py",
        revision=revision,
        label=label,
        function_name="target",
        qualified_name="target",
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
        make_sample(
            sample_id=pair_id + "-v",
            pair_id=pair_id,
            revision="vulnerable",
            label=1,
        ),
        make_sample(
            sample_id=pair_id + "-f",
            pair_id=pair_id,
            revision="fixed",
            label=0,
        ),
    ]


class FakeFunctionSampleBuilder:
    """Treat each supplied sample as one extracted function."""

    def build(self, sample):
        return [sample]


class FakePairExtractor:
    """Preserve the deterministic pairs supplied by the test."""

    def __init__(self):
        self.audit = {"counts": {}}

    def extract(self, samples):
        return samples


class FakeDiffLocalizer:
    def localize(self, sample):
        return sample.seed_lines


class FakeCFGBuilder:
    def __init__(self, failing_sample_ids=None):
        self.failing_sample_ids = set(failing_sample_ids or [])

    def build(self, sample):
        if sample.sample_id in self.failing_sample_ids:
            return None

        return {
            "nodes": [
                SimpleNamespace(
                    node_id=0,
                    lineno=1,
                    end_lineno=3,
                )
            ],
            "edges": [],
        }


class FakeCFGLocalizer:
    def __init__(self, empty_seed_sample_ids=None):
        self.empty_seed_sample_ids = set(
            empty_seed_sample_ids or []
        )

    def localize(self, sample):
        if sample.sample_id in self.empty_seed_sample_ids:
            sample.seed_nodes = []
        else:
            sample.seed_nodes = [0]

        return sample


class FakeFunctionScopeLocalizer:
    def __init__(self, empty_scope_sample_ids=None):
        self.empty_scope_sample_ids = set(
            empty_scope_sample_ids or []
        )

    def localize(self, sample):
        if sample.sample_id in self.empty_scope_sample_ids:
            sample.function_nodes = []
        else:
            sample.function_nodes = [0]

        return sample


class FakePruner:
    def __init__(self, empty_pruned_sample_ids=None):
        self.empty_pruned_sample_ids = set(
            empty_pruned_sample_ids or []
        )

    def prune(self, sample):
        if sample.sample_id in self.empty_pruned_sample_ids:
            sample.pruned_cfg = {
                "nodes": [],
                "edges": [],
            }
        else:
            sample.pruned_cfg = {
                "nodes": [{"id": 0}],
                "edges": [],
            }

        return sample


def make_pipeline(
    failing_cfg_ids=None,
    empty_seed_ids=None,
    empty_scope_ids=None,
    empty_pruned_ids=None,
):
    return FunctionDatasetPreparationPipeline(
        function_sample_builder=FakeFunctionSampleBuilder(),
        paired_function_extractor=FakePairExtractor(),
        cfg_builder=FakeCFGBuilder(failing_cfg_ids),
        diff_localizer=FakeDiffLocalizer(),
        cfg_localizer=FakeCFGLocalizer(empty_seed_ids),
        function_scope_localizer=FakeFunctionScopeLocalizer(
            empty_scope_ids
        ),
        pruner=FakePruner(empty_pruned_ids),
    )


def run_pipeline(pipeline, samples):
    """Run preparation without relying on console output."""
    return pipeline.prepare(samples)


def sample_ids(samples):
    return {sample.sample_id for sample in samples}


def pair_ids(samples):
    return {
        sample.function_pair_id
        for sample in samples
        if sample.function_pair_id
    }


def assert_pair_complete(samples, expected_pair_ids):
    """Every retained pair must have one vulnerable and one fixed sample."""
    groups = {}

    for sample in samples:
        groups.setdefault(sample.function_pair_id, []).append(sample)

    assert set(groups) == expected_pair_ids

    for pair_id, members in groups.items():
        assert len(members) == 2, pair_id
        assert {member.label for member in members} == {0, 1}
        assert {
            member.revision for member in members
        } == {"vulnerable", "fixed"}


def audit_events(pipeline, reason=None, stage=None):
    events = pipeline.audit["events"]

    if reason is not None:
        events = [
            event for event in events
            if event["reason"] == reason
        ]

    if stage is not None:
        events = [
            event for event in events
            if event["stage"] == stage
        ]

    return events


# ---------------------------------------------------------
# Scenario 1: both functions survive every stage
# ---------------------------------------------------------

def test_pipeline_keeps_successful_vulnerable_fixed_pair():
    samples = make_pair("success")
    pipeline = make_pipeline()

    prepared = run_pipeline(pipeline, samples)

    assert sample_ids(prepared) == {
        "success-v",
        "success-f",
    }
    assert pair_ids(prepared) == {"success"}
    assert_pair_complete(prepared, {"success"})

    assert not audit_events(pipeline, reason="cfg_build_failed")
    assert not audit_events(pipeline, reason="no_seed_nodes")
    assert not audit_events(
        pipeline, reason="no_valid_function_seed_nodes"
    )
    assert not audit_events(
        pipeline, reason="pruned_cfg_is_none"
    )
    assert not audit_events(
        pipeline, reason="pruned_cfg_has_no_nodes"
    )


# ---------------------------------------------------------
# Scenario 2: vulnerable member fails CFG construction
# ---------------------------------------------------------

def test_pipeline_excludes_pair_when_cfg_construction_fails():
    samples = make_pair("cfg-failure")
    pipeline = make_pipeline(
        failing_cfg_ids={"cfg-failure-v"}
    )

    prepared = run_pipeline(pipeline, samples)

    assert prepared == []
    assert pair_ids(prepared) == set()

    failures = audit_events(
        pipeline,
        reason="cfg_build_failed",
        stage="cfg_construction",
    )

    assert any(
        event["sample_id"] == "cfg-failure-v"
        and event["pair_id"] == "cfg-failure"
        for event in failures
    )

    pair_exclusions = audit_events(
        pipeline,
        reason="incomplete_pair",
        stage="cfg_construction",
    )

    assert any(
        event["pair_id"] == "cfg-failure"
        for event in pair_exclusions
    )


# ---------------------------------------------------------
# Scenario 3: fixed member has no localized seed nodes
# ---------------------------------------------------------

def test_pipeline_excludes_pair_when_fixed_has_no_seed_nodes():
    samples = make_pair("empty-seeds")
    pipeline = make_pipeline(
        empty_seed_ids={"empty-seeds-f"}
    )

    prepared = run_pipeline(pipeline, samples)

    assert prepared == []
    assert pair_ids(prepared) == set()

    failures = audit_events(
        pipeline,
        reason="no_seed_nodes",
        stage="cfg_localization",
    )

    assert any(
        event["sample_id"] == "empty-seeds-f"
        and event["pair_id"] == "empty-seeds"
        for event in failures
    )

    pair_exclusions = audit_events(
        pipeline,
        reason="incomplete_pair",
        stage="cfg_localization",
    )

    assert any(
        event["pair_id"] == "empty-seeds"
        for event in pair_exclusions
    )


# ---------------------------------------------------------
# Scenario 4: function-scope localization removes all seeds
# ---------------------------------------------------------

def test_pipeline_excludes_pair_when_function_scope_has_no_seeds():
    samples = make_pair("empty-scope")
    pipeline = make_pipeline(
        empty_scope_ids={"empty-scope-f"}
    )

    prepared = run_pipeline(pipeline, samples)

    assert prepared == []
    assert pair_ids(prepared) == set()

    failures = audit_events(
        pipeline,
        reason="no_valid_function_seed_nodes",
        stage="function_scope",
    )

    assert any(
        event["sample_id"] == "empty-scope-f"
        and event["pair_id"] == "empty-scope"
        for event in failures
    )

    pair_exclusions = audit_events(
        pipeline,
        reason="incomplete_pair",
        stage="function_scope",
    )

    assert any(
        event["pair_id"] == "empty-scope"
        for event in pair_exclusions
    )


# ---------------------------------------------------------
# Scenario 5: pruning produces an empty CFG
# ---------------------------------------------------------

def test_pipeline_excludes_pair_when_pruned_cfg_is_empty():
    samples = make_pair("empty-pruned")
    pipeline = make_pipeline(
        empty_pruned_ids={"empty-pruned-f"}
    )

    prepared = run_pipeline(pipeline, samples)

    assert prepared == []
    assert pair_ids(prepared) == set()

    failures = audit_events(
        pipeline,
        reason="pruned_cfg_has_no_nodes",
        stage="pruning",
    )

    assert any(
        event["sample_id"] == "empty-pruned-f"
        and event["pair_id"] == "empty-pruned"
        for event in failures
    )

    pair_exclusions = audit_events(
        pipeline,
        reason="incomplete_pair",
        stage="pruning",
    )

    assert any(
        event["pair_id"] == "empty-pruned"
        for event in pair_exclusions
    )


# ---------------------------------------------------------
# Scenario 6: pair arrives with only one member
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "revision,label",
    [
        ("vulnerable", 1),
        ("fixed", 0),
    ],
)
def test_pipeline_rejects_pair_missing_a_member(revision, label):
    sample = make_sample(
        sample_id="orphan-" + revision,
        pair_id="orphan-pair",
        revision=revision,
        label=label,
    )

    pipeline = make_pipeline()
    prepared = run_pipeline(pipeline, [sample])

    assert prepared == []
    assert pair_ids(prepared) == set()

    exclusions = audit_events(
        pipeline,
        reason="incomplete_pair",
        stage="pairing",
    )

    assert any(
        event["pair_id"] == "orphan-pair"
        and event["details"]["member_count"] == 1
        for event in exclusions
    )
