
from types import SimpleNamespace

from src.cfg.cfg_builder import CFGBuilder
from src.localizer.LineCFGLocalizer import LineCFGLocalizer


def test_line_localizer_handles_empty_seed_lines():
    sample = SimpleNamespace(
        cfg={
            "nodes": [
                SimpleNamespace(
                    node_id=0,
                    lineno=1,
                    end_lineno=3,
                )
            ]
        },
        seed_lines=[],
        line_to_node={},
        seed_nodes=[],
    )

    result = LineCFGLocalizer().localize(sample)

    assert result is sample
    assert result.seed_nodes == []
    assert result.line_to_node == {}


def test_cfg_builder_builds_whole_file():
    source = """
def target_function(value):
    return value

def unrelated_function():
    return 42
"""

    sample = SimpleNamespace(source=source)
    cfg = CFGBuilder().build(sample)

    assert cfg is not None
    assert "nodes" in cfg
    assert "edges" in cfg

    function_nodes = [
        node.text
        for node in cfg["nodes"]
        if node.node_type in ("FunctionDef", "AsyncFunctionDef")
    ]

    assert "def target_function" in function_nodes
    assert "def unrelated_function" in function_nodes


def test_cfg_builder_returns_none_for_syntax_error():
    sample = SimpleNamespace(
        source="def broken_function(:\n    pass\n"
    )

    assert CFGBuilder().build(sample) is None

from types import SimpleNamespace

from src.pipeline.function_dataset_preparation_pipeline import (
    FunctionDatasetPreparationPipeline,
)


def make_sample(sample_id, pair_id, revision, label):
    return SimpleNamespace(
        sample_id=sample_id,
        function_pair_id=pair_id,
        parent_sample_id=None,
        repo="test-repo",
        fix_commit_hash="fix-123",
        commit_hash="commit-" + revision,
        file_path="example.py",
        revision=revision,
        label=label,
        function_name="target",
        qualified_name="target",
    )


def test_pair_validation_rejects_incomplete_pairs():
    pipeline = FunctionDatasetPreparationPipeline(
        function_sample_builder=None,
        paired_function_extractor=None,
        cfg_builder=None,
        diff_localizer=None,
        cfg_localizer=None,
        function_scope_localizer=None,
        pruner=None,
    )

    complete_vulnerable = make_sample(
        "sample-v1", "pair-complete", "vulnerable", 1
    )
    complete_fixed = make_sample(
        "sample-f1", "pair-complete", "fixed", 0
    )

    missing_fixed = make_sample(
        "sample-v2", "pair-missing-fixed", "vulnerable", 1
    )

    missing_vulnerable = make_sample(
        "sample-f2", "pair-missing-vulnerable", "fixed", 0
    )

    samples = [
        complete_vulnerable,
        complete_fixed,
        missing_fixed,
        missing_vulnerable,
    ]

    retained = pipeline._retain_complete_pairs(
        samples, "test_pair_validation"
    )

    assert {s.sample_id for s in retained} == {
        "sample-v1",
        "sample-f1",
    }

    assert {
        s.function_pair_id for s in retained
    } == {"pair-complete"}

    assert pipeline.audit["counts"][
        "test_pair_validation_pairs_dropped"
    ] == 2

def test_audit_event_contains_correct_metadata():
    pipeline = FunctionDatasetPreparationPipeline(
        function_sample_builder=None,
        paired_function_extractor=None,
        cfg_builder=None,
        diff_localizer=None,
        cfg_localizer=None,
        function_scope_localizer=None,
        pruner=None,
    )

    sample = make_sample(
        sample_id="sample-001",
        pair_id="pair-001",
        revision="vulnerable",
        label=1,
    )

    pipeline._record_event(
        stage="cfg_construction",
        reason="cfg_build_failed",
        sample=sample,
    )

    assert len(pipeline.audit["events"]) == 1

    event = pipeline.audit["events"][0]

    assert event["sample_id"] == "sample-001"
    assert event["pair_id"] == "pair-001"
    assert event["stage"] == "cfg_construction"
    assert event["reason"] == "cfg_build_failed"