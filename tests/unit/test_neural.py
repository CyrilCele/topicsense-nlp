"""Tests for neural text classification components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from topicsense.models.neural import BiLSTM, TextCNN
from topicsense.training.neural import (
    build_vocabulary,
    encode_documents,
    run_neural_experiments,
)


def test_textcnn_returns_class_scores() -> None:
    """TextCNN should return one score per class and input document."""
    model = TextCNN(
        vocabulary_size=20,
        num_classes=3,
        embedding_dim=8,
        num_filters=4,
        kernel_sizes=(2, 3),
        dropout=0.1,
    )

    input_ids = torch.randint(1, 20, (5, 8))
    output = model(input_ids)

    assert output.shape == (5, 3)


def test_textcnn_rejects_short_sequences() -> None:
    """Input sequences must be long enough for the largest convolution."""
    model = TextCNN(
        vocabulary_size=20,
        num_classes=3,
        embedding_dim=8,
        num_filters=4,
        kernel_sizes=(3, 4),
    )

    with pytest.raises(ValueError, match="sequence length"):
        model(torch.ones((2, 3), dtype=torch.long))


def test_build_vocabulary_is_deterministic() -> None:
    """Vocabulary construction should be stable for the same training text."""
    documents = [
        "space science astronomy",
        "space science telescope",
        "football sports stadium",
    ]

    first = build_vocabulary(
        documents,
        max_size=20,
        min_frequency=1,
    )
    second = build_vocabulary(
        documents,
        max_size=20,
        min_frequency=1,
    )

    assert first == second
    assert first["<PAD>"] == 0
    assert first["<UNK>"] == 1
    assert "space" in first


def test_encode_documents_pads_and_truncates() -> None:
    """Document encoding should apply the configured sequence length."""
    vocabulary = {
        "<PAD>": 0,
        "<UNK>": 1,
        "space": 2,
        "science": 3,
        "telescope": 4,
    }

    encoded = encode_documents(
        ["space science telescope", "space unknown"],
        vocabulary,
        max_sequence_length=2,
    )

    assert encoded.shape == (2, 2)
    assert encoded[0].tolist() == [2, 3]
    assert encoded[1].tolist() == [2, 1]


def test_neural_runner_uses_training_split_only(tmp_path: Path) -> None:
    """The neural runner must exclude official test records."""
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
                "validation_record_ids": [
                    "train-5",
                    "train-6",
                ],
            }
        ),
        encoding="utf-8",
    )

    config: dict[str, Any] = {
        "project": {"random_seed": 42},
        "data": {
            "raw_dataset_path": "dataset.jsonl",
            "split_manifest_path": "split.json",
        },
        "preprocessing": {
            "representation": "message_body",
            "lowercase": True,
        },
        "vocabulary": {
            "max_size": 100,
            "min_frequency": 1,
        },
        "model": {
            "embedding_dim": 8,
            "num_filters": 4,
            "kernel_sizes": [2, 3],
            "dropout": 0.1,
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "patience": 2,
            "max_grad_norm": 1.0,
            "max_sequence_length": 8,
            "num_workers": 0,
            "num_threads": 1,
            "save_model": False,
            "artifact_dir": "models/checkpoints/neural",
        },
        "outputs": {
            "metrics_csv": "results/metrics/test_neural.csv",
            "results_json": "results/logs/test_neural.json",
        },
    }

    result = run_neural_experiments(config, project_root=tmp_path)

    assert len(result["runs"]) == 1
    assert result["runs"][0]["training_documents"] == 4
    assert result["runs"][0]["validation_documents"] == 2
    assert result["runs"][0]["experiment_id"] == "EXP-04"

    payload = json.loads(
        (tmp_path / "results/logs/test_neural.json").read_text(encoding="utf-8")
    )

    assert payload["test_data_used"] is False
    assert payload["training_documents"] == 4
    assert payload["validation_documents"] == 2
    assert (tmp_path / "results/metrics/test_neural.csv").is_file()


def test_bilstm_returns_class_scores() -> None:
    """BiLSTM should return one score per class and input document."""

    model = BiLSTM(
        vocabulary_size=20,
        num_classes=3,
        embedding_dim=8,
        hidden_size=4,
        num_layers=1,
        dropout=0.1,
    )

    input_ids = torch.tensor(
        [
            [2, 3, 4, 0, 0],
            [5, 6, 7, 8, 0],
            [9, 10, 11, 12, 13],
        ],
        dtype=torch.long,
    )

    output = model(input_ids)

    assert output.shape == (3, 3)


def test_bilstm_handles_empty_token_sequences() -> None:
    """BiLSTM should handle sequences containing only padding tokens."""

    model = BiLSTM(
        vocabulary_size=20,
        num_classes=3,
        embedding_dim=8,
        hidden_size=4,
        num_layers=1,
        dropout=0.1,
    )

    input_ids = torch.zeros((2, 5), dtype=torch.long)

    output = model(input_ids)

    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
