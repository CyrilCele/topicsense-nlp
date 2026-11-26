"""Tests for consolidated model comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from topicsense.evaluation.comparison import (
    build_model_comparison,
    write_model_comparison,
)

CLASS_LABELS = ["science", "sports"]


def _metrics(macro_f1: float) -> dict[str, Any]:
    return {
        "accuracy": 0.75,
        "macro_precision": 0.75,
        "macro_recall": 0.75,
        "macro_f1": macro_f1,
        "weighted_f1": 0.75,
        "class_labels": CLASS_LABELS,
        "per_class": {
            "science": {
                "precision": 0.75,
                "recall": 0.75,
                "f1_score": 0.75,
                "support": 2,
            },
            "sports": {
                "precision": 0.75,
                "recall": 0.75,
                "f1_score": 0.75,
                "support": 2,
            },
        },
        "confusion_matrix": [[2, 0], [1, 1]],
    }


def _payload(
    *,
    training_documents: int = 4,
    validation_documents: int = 2,
) -> dict[str, Any]:
    return {
        "dataset": "20newsgroups",
        "development_split": "split.json",
        "random_seed": 42,
        "training_documents": training_documents,
        "validation_documents": validation_documents,
        "class_labels": CLASS_LABELS,
        "primary_selection_metric": "macro_f1",
        "test_data_used": False,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _create_result_files(tmp_path: Path) -> dict[str, str]:
    classical = _payload()
    classical["runs"] = [
        {
            "run_id": "naive_bayes__count__message_body",
            "experiment_id": "EXP-01",
            "classifier": "multinomial_nb",
            "feature_type": "count",
            "representation": "message_body",
            "training_documents": 4,
            "validation_documents": 2,
            "fit_seconds": 1.0,
            "prediction_seconds": 0.1,
            "latency_ms_per_document": 50.0,
            "throughput_documents_per_second": 20.0,
            "artifact_size_bytes": 1000,
            "metrics": _metrics(0.70),
        }
    ]

    textcnn = _payload()
    textcnn["runs"] = [
        {
            "run_id": "textcnn__message_body",
            "experiment_id": "EXP-04",
            "model": "TextCNN",
            "training_documents": 4,
            "validation_documents": 2,
            "fit_seconds": 2.0,
            "prediction_seconds": 0.2,
            "latency_ms_per_document": 100.0,
            "throughput_documents_per_second": 10.0,
            "artifact_size_bytes": 2000,
            "metrics": _metrics(0.80),
        }
    ]

    bilstm = _payload()
    bilstm["runs"] = [
        {
            "run_id": "bilstm__message_body",
            "experiment_id": "EXP-05",
            "model": "BiLSTM",
            "training_documents": 4,
            "validation_documents": 2,
            "fit_seconds": 3.0,
            "prediction_seconds": 0.3,
            "latency_ms_per_document": 150.0,
            "throughput_documents_per_second": 6.67,
            "artifact_size_bytes": 3000,
            "metrics": _metrics(0.65),
        }
    ]

    distilbert = _payload()
    distilbert.pop("training_documents")
    distilbert.pop("validation_documents")
    distilbert["summary"] = {
        "run_id": "distilbert__message_body",
        "experiment_id": "EXP-06",
        "training_documents": 4,
        "validation_documents": 2,
        "fit_seconds": 10.0,
        "end_to_end_inference_seconds": 0.5,
        "end_to_end_latency_ms_per_document": 250.0,
        "end_to_end_throughput_documents_per_second": 4.0,
        "model_and_tokenizer_artifact_size_bytes": 4000,
    }
    distilbert["metrics"] = _metrics(0.75)

    frozen = _payload()
    frozen.pop("training_documents")
    frozen.pop("validation_documents")
    frozen["summary"] = {
        "run_id": "frozen_minilm__message_body",
        "experiment_id": "EXP-07",
        "training_documents": 4,
        "validation_documents": 2,
        "training_embedding_seconds": 4.0,
        "fit_seconds": 1.0,
        "end_to_end_inference_seconds": 0.4,
        "end_to_end_latency_ms_per_document": 200.0,
        "end_to_end_throughput_documents_per_second": 5.0,
        "classifier_artifact_size_bytes": 500,
        "peak_sampled_rss_bytes": 100000,
    }
    frozen["metrics"] = _metrics(0.77)

    payloads = {
        "classical": classical,
        "textcnn": textcnn,
        "bilstm": bilstm,
        "distilbert": distilbert,
        "frozen": frozen,
    }

    paths = {}

    for name, payload in payloads.items():
        relative_path = f"results/logs/{name}.json"
        _write_json(tmp_path / relative_path, payload)
        paths[name] = relative_path

    return paths


def _comparison_config(paths: dict[str, str]) -> dict[str, Any]:
    return {
        "project": {"random_seed": 42},
        "data": {
            "dataset": "20newsgroups",
            "development_split": "split.json",
            "expected_training_documents": 4,
            "expected_validation_documents": 2,
            "expected_class_count": 2,
        },
        "inputs": {
            "classical_results": paths["classical"],
            "textcnn_results": paths["textcnn"],
            "bilstm_results": paths["bilstm"],
            "distilbert_results": paths["distilbert"],
            "frozen_embeddings_results": paths["frozen"],
        },
        "comparison": {
            "primary_metric": "macro_f1",
            "require_test_data_excluded": True,
            "expected_classical_run_count": 1,
            "expected_total_runs": 5,
        },
        "outputs": {
            "comparison_metrics_csv": "results/metrics/comparison.csv",
            "comparison_results_json": "results/logs/comparison.json",
        },
    }


def test_build_model_comparison_consolidates_all_families(
    tmp_path: Path,
) -> None:
    paths = _create_result_files(tmp_path)
    config = _comparison_config(paths)

    result = build_model_comparison(
        config,
        project_root=tmp_path,
    )

    assert result["metadata"]["experiment_count"] == 5
    assert len(result["runs"]) == 5
    assert result["best_overall_validation"]["model"] == "TextCNN"
    assert len(result["best_by_family"]) == 3
    assert result["metadata"]["test_data_used"] is False


def test_build_model_comparison_rejects_test_data_usage(
    tmp_path: Path,
) -> None:
    paths = _create_result_files(tmp_path)

    classical_path = tmp_path / paths["classical"]
    classical = json.loads(classical_path.read_text(encoding="utf-8"))
    classical["test_data_used"] = True
    classical_path.write_text(
        json.dumps(classical),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="test data was excluded"):
        build_model_comparison(
            _comparison_config(paths),
            project_root=tmp_path,
        )


def test_build_model_comparison_rejects_inconsistent_validation_counts(
    tmp_path: Path,
) -> None:
    paths = _create_result_files(tmp_path)

    textcnn_path = tmp_path / paths["textcnn"]
    textcnn = json.loads(textcnn_path.read_text(encoding="utf-8"))
    textcnn["validation_documents"] = 3
    textcnn_path.write_text(
        json.dumps(textcnn),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="validation-document count"):
        build_model_comparison(
            _comparison_config(paths),
            project_root=tmp_path,
        )


def test_write_model_comparison_creates_outputs(
    tmp_path: Path,
) -> None:
    paths = _create_result_files(tmp_path)
    config = _comparison_config(paths)

    comparison = build_model_comparison(
        config,
        project_root=tmp_path,
    )

    csv_path, json_path = write_model_comparison(
        comparison,
        config=config,
        project_root=tmp_path,
    )

    assert csv_path.is_file()
    assert json_path.is_file()

    saved = json.loads(json_path.read_text(encoding="utf-8"))

    assert saved["metadata"]["experiment_count"] == 5
    assert len(saved["runs"]) == 5
