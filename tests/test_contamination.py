"""Tests for train-test contamination manifest generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from topicsense.data.contamination import build_contamination_manifests


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON fixture."""
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL fixtures."""
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def body_hash(body: str) -> str:
    """Calculate the SHA-256 digest of a normalized body."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def make_record(
    record_id: str,
    source_split: str,
    label: str,
    body: str,
) -> dict[str, str]:
    """Create a minimal raw dataset record."""
    return {
        "record_id": record_id,
        "source_split": source_split,
        "label": label,
        "text": f"Subject: Test\n\n{body}",
    }


def create_test_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    """Create a synthetic dataset and audit reports."""
    raw_path = tmp_path / "raw.jsonl"
    body_audit_path = tmp_path / "body_audit.json"
    leakage_path = tmp_path / "leakage.json"
    output_dir = tmp_path / "output"

    substantive_body = (
        "This is a sufficiently long repeated document body "
        "used to test duplicate detection."
    )
    trivial_body = "test"

    records = [
        make_record(
            "train-001",
            "train",
            "topic_a",
            substantive_body,
        ),
        make_record(
            "test-001",
            "test",
            "topic_a",
            substantive_body,
        ),
        make_record(
            "test-002",
            "test",
            "topic_a",
            substantive_body,
        ),
        make_record(
            "train-002",
            "train",
            "topic_b",
            trivial_body,
        ),
        make_record(
            "test-003",
            "test",
            "topic_c",
            trivial_body,
        ),
        make_record(
            "train-003",
            "train",
            "topic_a",
            "A separate training document.",
        ),
        make_record(
            "test-004",
            "test",
            "topic_a",
            "A separate test document.",
        ),
    ]
    write_jsonl(raw_path, records)

    body_audit = {
        "duplicate_groups": [
            {
                "body_hash": body_hash(substantive_body),
                "cross_partition": True,
                "label_agreement": True,
                "members": [
                    {"record_id": "train-001"},
                    {"record_id": "test-001"},
                    {"record_id": "test-002"},
                ],
            },
            {
                "body_hash": body_hash(trivial_body),
                "cross_partition": True,
                "label_agreement": False,
                "members": [
                    {"record_id": "train-002"},
                    {"record_id": "test-003"},
                ],
            },
        ]
    }
    write_json(body_audit_path, body_audit)

    leakage_report = {
        "duplicate_analysis": {
            "near_duplicate_pairs": [
                {"cross_partition": True},
                {"cross_partition": False},
            ]
        }
    }
    write_json(leakage_path, leakage_report)

    return raw_path, body_audit_path, leakage_path, output_dir


def test_builds_expected_sensitivity_manifests(
    tmp_path: Path,
) -> None:
    """Substantive duplicates are excluded; trivial matches are retained."""
    raw_path, body_audit_path, leakage_path, output_dir = create_test_inputs(tmp_path)

    manifest = build_contamination_manifests(
        raw_path=raw_path,
        body_audit_path=body_audit_path,
        leakage_report_path=leakage_path,
        output_dir=output_dir,
    )

    summary = manifest["summary"]

    assert summary["substantive_cross_partition_exact_duplicate_groups"] == 1
    assert summary["substantive_training_documents_implicated"] == 1
    assert summary["substantive_test_documents_implicated"] == 2
    assert summary["trivial_cross_partition_exact_duplicate_groups"] == 1
    assert summary["cross_partition_near_duplicate_pairs_not_excluded"] == 1
    assert summary["clean_test_document_count"] == 2
    assert summary["decontaminated_training_document_count"] == 2

    clean_test = json.loads(
        (output_dir / "clean_test_ids.json").read_text(encoding="utf-8")
    )
    clean_train = json.loads(
        (output_dir / "decontaminated_train_ids.json").read_text(encoding="utf-8")
    )

    assert clean_test["record_ids"] == ["test-001", "test-002"]
    assert clean_train["record_ids"] == ["train-001"]


def test_rejects_duplicate_record_ids(
    tmp_path: Path,
) -> None:
    """Duplicate raw record IDs must not be accepted."""
    raw_path, body_audit_path, leakage_path, output_dir = create_test_inputs(tmp_path)

    records = [
        make_record("duplicate-id", "train", "topic_a", "First document."),
        make_record("duplicate-id", "test", "topic_a", "Second document."),
    ]
    write_jsonl(raw_path, records)

    with pytest.raises(ValueError, match="Duplicate record_id"):
        build_contamination_manifests(
            raw_path=raw_path,
            body_audit_path=body_audit_path,
            leakage_report_path=leakage_path,
            output_dir=output_dir,
        )


def test_rejects_inconsistent_duplicate_group(
    tmp_path: Path,
) -> None:
    """A group with different normalized bodies must be rejected."""
    raw_path, body_audit_path, leakage_path, output_dir = create_test_inputs(tmp_path)

    body_audit = json.loads(body_audit_path.read_text(encoding="utf-8"))
    body_audit["duplicate_groups"][0]["members"][1]["record_id"] = "test-004"
    write_json(body_audit_path, body_audit)

    with pytest.raises(ValueError, match="different normalized bodies"):
        build_contamination_manifests(
            raw_path=raw_path,
            body_audit_path=body_audit_path,
            leakage_report_path=leakage_path,
            output_dir=output_dir,
        )
