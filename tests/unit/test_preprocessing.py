"""Unit tests for reusable preprocessing utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from topicsense.preprocessing.splitting import create_preprocessing_splits
from topicsense.preprocessing.text import (
    extract_message_body,
    normalize_text,
    preprocess_text,
    tokenize_words,
)


def test_extract_message_body_preserves_body_content() -> None:
    text = "Subject: Example\nFrom: user@example.com\n\nHello, World!\n> quoted text"
    assert extract_message_body(text) == "Hello, World!\n> quoted text"


def test_extract_message_body_falls_back_to_full_text() -> None:
    assert extract_message_body("No headers here") == "No headers here"


def test_normalize_text_is_conservative() -> None:
    assert normalize_text("  Ｈello   WORLD!\n") == "Hello WORLD!"
    assert normalize_text("  Ｈello   WORLD!\n", lowercase=True) == "hello world!"


def test_preprocess_text_requires_known_representation() -> None:
    with pytest.raises(ValueError, match="representation"):
        preprocess_text("text", representation="unknown")


def test_tokenize_words_is_diagnostic_and_keeps_apostrophes() -> None:
    assert tokenize_words("Don't remove topic-specific words.") == [
        "Don't",
        "remove",
        "topic",
        "specific",
        "words",
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_split_keeps_duplicate_connected_documents_together(
    tmp_path: Path,
) -> None:
    records = []

    for label_index, label in enumerate(("topic_a", "topic_b", "topic_c")):
        for index in range(10):
            records.append(
                {
                    "record_id": f"train-{label_index}-{index}",
                    "source_split": "train",
                    "label": label,
                    "text": f"Subject: sample\n\nDocument {label} {index}",
                }
            )

    records.append(
        {
            "record_id": "test-0",
            "source_split": "test",
            "label": "topic_a",
            "text": "Held-out test message",
        }
    )

    raw_path = tmp_path / "raw.jsonl"
    body_audit_path = tmp_path / "body_audit.json"
    leakage_path = tmp_path / "leakage.json"
    exclusions_path = tmp_path / "decontaminated.json"
    output_dir = tmp_path / "splits"

    _write_jsonl(raw_path, records)

    _write_json(
        body_audit_path,
        {
            "duplicate_groups": [
                {
                    "members": [
                        {"record_id": "train-0-0"},
                        {"record_id": "train-0-1"},
                    ]
                }
            ]
        },
    )

    _write_json(
        leakage_path,
        {
            "duplicate_analysis": {
                "near_duplicate_pairs": [
                    {
                        "left_record_id": "train-1-0",
                        "right_record_id": "train-1-1",
                    },
                    {
                        "left_record_id": "train-0-0",
                        "right_record_id": "test-0",
                    },
                ]
            }
        },
    )

    _write_json(exclusions_path, {"record_ids": ["train-2-9"]})

    result = create_preprocessing_splits(
        raw_dataset_path=raw_path,
        body_audit_path=body_audit_path,
        leakage_report_path=leakage_path,
        decontaminated_train_ids_path=exclusions_path,
        output_dir=output_dir,
        validation_size=0.20,
        random_seed=42,
    )

    split = result["splits"]["official_training"]
    training_ids = set(split["training_record_ids"])
    validation_ids = set(split["validation_record_ids"])

    assert not (training_ids & validation_ids)

    assert {"train-0-0", "train-0-1"} <= training_ids or {
        "train-0-0",
        "train-0-1",
    } <= validation_ids

    assert {"train-1-0", "train-1-1"} <= training_ids or {
        "train-1-0",
        "train-1-1",
    } <= validation_ids

    assert "test-0" not in training_ids | validation_ids
    assert split["group_overlap_count"] == 0

    decontaminated_split = result["splits"]["decontaminated_training"]

    assert decontaminated_split["document_count"] == 29

    assert "train-2-9" not in (
        set(decontaminated_split["training_record_ids"])
        | set(decontaminated_split["validation_record_ids"])
    )

    assert (output_dir / "preprocessing_split_manifest.json").is_file()


def test_split_rejects_non_reciprocal_validation_fraction(
    tmp_path: Path,
) -> None:
    records = [
        {
            "record_id": f"train-{index}",
            "source_split": "train",
            "label": "topic_a",
            "text": f"Document {index}",
        }
        for index in range(10)
    ]

    raw_path = tmp_path / "raw.jsonl"
    body_audit_path = tmp_path / "body_audit.json"
    leakage_path = tmp_path / "leakage.json"
    exclusions_path = tmp_path / "decontaminated.json"

    _write_jsonl(raw_path, records)
    _write_json(body_audit_path, {"duplicate_groups": []})

    _write_json(
        leakage_path,
        {"duplicate_analysis": {"near_duplicate_pairs": []}},
    )

    _write_json(exclusions_path, {"record_ids": []})

    with pytest.raises(ValueError, match="reciprocal"):
        create_preprocessing_splits(
            raw_dataset_path=raw_path,
            body_audit_path=body_audit_path,
            leakage_report_path=leakage_path,
            decontaminated_train_ids_path=exclusions_path,
            output_dir=tmp_path / "splits",
            validation_size=0.15,
        )
