"""Tests for transformer experiment utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from topicsense.training.transformers import (
    _load_development_data,
    run_transformer_experiments,
)


def _write_dataset_files(tmp_path: Path) -> tuple[Path, Path]:
    raw_path = tmp_path / "dataset.jsonl"
    split_path = tmp_path / "split.json"

    records = [
        {
            "record_id": "train-1",
            "source_split": "train",
            "text": "space astronomy telescope",
            "label": "science",
        },
        {
            "record_id": "train-2",
            "source_split": "train",
            "text": "space rocket orbit",
            "label": "science",
        },
        {
            "record_id": "train-3",
            "source_split": "train",
            "text": "football goal stadium",
            "label": "sports",
        },
        {
            "record_id": "train-4",
            "source_split": "train",
            "text": "football player match",
            "label": "sports",
        },
        {
            "record_id": "train-5",
            "source_split": "train",
            "text": "planet telescope galaxy",
            "label": "science",
        },
        {
            "record_id": "train-6",
            "source_split": "train",
            "text": "team football league",
            "label": "sports",
        },
        {
            "record_id": "test-1",
            "source_split": "test",
            "text": "this official test record must not be used",
            "label": "science",
        },
    ]

    raw_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    split_path.write_text(
        json.dumps(
            {
                "training_record_ids": [
                    "train-1",
                    "train-2",
                    "train-3",
                    "train-4",
                ],
                "validation_record_ids": ["train-5", "train-6"],
            }
        ),
        encoding="utf-8",
    )

    return raw_path, split_path


def test_transformer_data_loader_excludes_official_test(
    tmp_path: Path,
) -> None:
    raw_path, split_path = _write_dataset_files(tmp_path)

    train_texts, train_labels, validation_texts, validation_labels = (
        _load_development_data(
            raw_path=raw_path,
            split_path=split_path,
        )
    )

    assert len(train_texts) == 4
    assert len(train_labels) == 4
    assert len(validation_texts) == 2
    assert len(validation_labels) == 2
    assert "this official test record must not be used" not in train_texts
    assert "this official test record must not be used" not in validation_texts


def test_transformer_data_loader_rejects_overlapping_ids(
    tmp_path: Path,
) -> None:
    raw_path, split_path = _write_dataset_files(tmp_path)

    split_path.write_text(
        json.dumps(
            {
                "training_record_ids": ["train-1", "train-2"],
                "validation_record_ids": ["train-2", "train-3"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlap"):
        _load_development_data(
            raw_path=raw_path,
            split_path=split_path,
        )


def test_transformer_data_loader_rejects_unknown_ids(
    tmp_path: Path,
) -> None:
    raw_path, split_path = _write_dataset_files(tmp_path)

    split_path.write_text(
        json.dumps(
            {
                "training_record_ids": ["train-1", "train-2"],
                "validation_record_ids": ["unknown-record"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not found"):
        _load_development_data(
            raw_path=raw_path,
            split_path=split_path,
        )


def test_transformer_runner_rejects_unknown_experiment(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="experiment must be"):
        run_transformer_experiments(
            {},
            project_root=tmp_path,
            experiment="unsupported",
        )
