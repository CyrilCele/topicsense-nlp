"""Tests for classical model and feature components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from topicsense.evaluation.metrics import calculate_classification_metrics
from topicsense.features.vectorization import build_vectorizer
from topicsense.models.classical.estimators import build_classifier
from topicsense.training.classical import run_classical_experiments


def test_build_count_vectorizer() -> None:
    """Count vectorizer should respect configured options."""
    vectorizer = build_vectorizer(
        "count",
        {
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 1.0,
            "lowercase": False,
        },
    )

    transformed = vectorizer.fit_transform(["space exploration", "space science"])

    assert transformed.shape[0] == 2
    assert vectorizer.lowercase is False
    assert vectorizer.ngram_range == (1, 2)


def test_build_tfidf_vectorizer() -> None:
    """TF-IDF vectorizer should construct a valid sparse representation."""
    vectorizer = build_vectorizer(
        "tfidf",
        {
            "ngram_range": [1, 1],
            "min_df": 1,
            "max_df": 1.0,
            "lowercase": False,
            "sublinear_tf": True,
        },
    )

    transformed = vectorizer.fit_transform(["space exploration", "computer graphics"])

    assert transformed.shape[0] == 2
    assert vectorizer.sublinear_tf is True


def test_unsupported_vectorizer_raises_error() -> None:
    """Unsupported feature types should fail explicitly."""
    with pytest.raises(ValueError, match="feature_type"):
        build_vectorizer("unsupported", {})


def test_build_classical_estimators() -> None:
    """All planned classical classifiers should be constructible."""
    assert build_classifier("multinomial_nb", {}).__class__.__name__ == (
        "MultinomialNB"
    )
    assert build_classifier("logistic_regression", {}).__class__.__name__ == (
        "LogisticRegression"
    )
    assert build_classifier("linear_svm", {}).__class__.__name__ == "LinearSVC"


def test_classification_metrics() -> None:
    """Evaluation should return aggregate and per-class metrics."""
    result = calculate_classification_metrics(
        ["science", "science", "sports", "sports"],
        ["science", "sports", "sports", "sports"],
        class_labels=["science", "sports"],
    )

    assert result["accuracy"] == 0.75
    assert result["macro_f1"] > 0.0
    assert set(result["per_class"]) == {"science", "sports"}
    assert len(result["confusion_matrix"]) == 2


def test_classical_runner_uses_training_split_only(tmp_path: Path) -> None:
    """The experiment runner should use only supplied training IDs."""
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
            "text": "this test record must not be used",
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
            "lowercase": False,
            "representations": ["message_body"],
        },
        "features": {
            "types": ["count"],
            "ngram_range": [1, 1],
            "min_df": 1,
            "max_df": 1.0,
            "max_features": 100,
            "lowercase": False,
        },
        "classifiers": {
            "multinomial_nb": {
                "enabled": True,
                "alpha": 1.0,
            }
        },
        "training": {
            "save_models": False,
            "artifact_dir": "models/checkpoints/classical_ml",
        },
        "outputs": {
            "metrics_csv": "results/metrics/test.csv",
            "results_json": "results/logs/test.json",
        },
    }

    result = run_classical_experiments(config, project_root=tmp_path)

    assert len(result["runs"]) == 1
    assert result["runs"][0]["training_documents"] == 4
    assert result["runs"][0]["validation_documents"] == 2

    payload = json.loads(
        (tmp_path / "results/logs/test.json").read_text(encoding="utf-8")
    )

    assert payload["test_data_used"] is False
    assert payload["training_documents"] == 4
    assert payload["validation_documents"] == 2
    assert (tmp_path / "results/metrics/test.csv").is_file()
