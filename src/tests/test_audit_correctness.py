
# File: src/tests/test_audit_correctness.py
# Action: CREATE

import json
from types import SimpleNamespace

from src.audit.DatasetAudit import DatasetAudit


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
        qualified_name="module.target",
    )


def test_stage_counts_count_samples_and_distinct_pairs():
    audit = DatasetAudit()

    samples = [
        make_sample("v1", "pair-1", "vulnerable", 1),
        make_sample("f1", "pair-1", "fixed", 0),
        make_sample("v2", "pair-2", "vulnerable", 1),
        make_sample("orphan", None, "vulnerable", 1),
    ]

    audit.record_stage("pairing", samples)

    stage = audit.report["stages"]["pairing"]

    assert stage["sample_count"] == 4
    assert stage["pair_count"] == 2
    assert stage["missing_pair_id_count"] == 1
    assert stage["label_counts"] == {"0": 1, "1": 3}


def test_pair_exclusion_records_remaining_members():
    audit = DatasetAudit()

    vulnerable = make_sample(
        "sample-v", "pair-1", "vulnerable", 1
    )

    audit.record_pair_exclusion(
        stage="cfg_construction",
        pair_id="pair-1",
        members=[vulnerable],
    )

    event = audit.report["events"][0]

    assert event["event_type"] == "pair_exclusion"
    assert event["stage"] == "cfg_construction"
    assert event["reason"] == "incomplete_pair"
    assert event["pair_id"] == "pair-1"
    assert event["details"]["member_count"] == 1
    assert event["details"]["sample_ids"] == ["sample-v"]
    assert audit.report["counts"][
        "cfg_construction:dropped_pairs"
    ] == 1


def test_missing_pair_id_is_separate_from_pair_drop_count():
    audit = DatasetAudit()
    sample = make_sample("orphan", None, "vulnerable", 1)

    audit.record_missing_pair_id("pairing", sample)

    assert audit.report["events"][0]["reason"] == "missing_pair_id"
    assert audit.report["events"][0]["sample_id"] == "orphan"
    assert "pairing:dropped_pairs" not in audit.report["counts"]


def test_sample_failure_and_pair_exclusion_are_distinct_events():
    audit = DatasetAudit()
    sample = make_sample("sample-v", "pair-1", "vulnerable", 1)

    audit.record_event(
        stage="cfg_construction",
        reason="cfg_build_failed",
        sample=sample,
    )
    audit.record_pair_exclusion(
        stage="cfg_construction",
        pair_id="pair-1",
        members=[],
    )

    assert len(audit.report["events"]) == 2
    assert audit.report["counts"][
        "cfg_construction:cfg_build_failed"
    ] == 1
    assert audit.report["counts"][
        "cfg_construction:dropped_pairs"
    ] == 1


def test_snapshot_is_independent():
    audit = DatasetAudit()
    audit.record_event("cfg_construction", "cfg_build_failed")

    snapshot = audit.snapshot()
    snapshot["events"].clear()
    snapshot["counts"].clear()

    assert len(audit.report["events"]) == 1
    assert audit.report["counts"][
        "cfg_construction:cfg_build_failed"
    ] == 1


def test_json_jsonl_and_csv_round_trip(tmp_path):
    audit = DatasetAudit()
    sample = make_sample("sample-v", "pair-1", "vulnerable", 1)

    audit.record_stage("raw", [sample])
    audit.record_event(
        stage="cfg_construction",
        reason="cfg_build_failed",
        sample=sample,
        details={"message": "test failure"},
    )

    json_path = str(tmp_path / "audit.json")
    jsonl_path = str(tmp_path / "events.jsonl")
    csv_path = str(tmp_path / "events.csv")

    audit.export_json(json_path)
    audit.export_jsonl(jsonl_path)
    audit.export_csv(csv_path)

    loaded_json = DatasetAudit.read_json(json_path)
    loaded_jsonl = DatasetAudit.read_jsonl(jsonl_path)
    loaded_csv = DatasetAudit.read_csv(csv_path)

    assert loaded_json["stages"]["raw"]["sample_count"] == 1
    assert loaded_json["events"][0]["sample_id"] == "sample-v"

    assert len(loaded_jsonl) == 1
    assert loaded_jsonl[0]["reason"] == "cfg_build_failed"

    assert len(loaded_csv) == 1
    assert loaded_csv[0]["sample_id"] == "sample-v"
    assert json.loads(loaded_csv[0]["details"]) == {
        "message": "test failure"
    }


def test_empty_reports_round_trip(tmp_path):
    audit = DatasetAudit()

    json_path = str(tmp_path / "empty.json")
    jsonl_path = str(tmp_path / "empty.jsonl")
    csv_path = str(tmp_path / "empty.csv")

    audit.export_json(json_path)
    audit.export_jsonl(jsonl_path)
    audit.export_csv(csv_path)

    assert DatasetAudit.read_json(json_path)["events"] == []
    assert DatasetAudit.read_jsonl(jsonl_path) == []
    assert DatasetAudit.read_csv(csv_path) == []
