"""Validation-result consolidation and model comparison utilities."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

METRIC_NAMES = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
)

CSV_FIELDS = (
    "rank",
    "run_id",
    "experiment_id",
    "model_family",
    "model",
    "representation",
    "feature_type",
    "training_documents",
    "validation_documents",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "training_related_seconds",
    "training_cost_scope",
    "end_to_end_inference_seconds",
    "end_to_end_latency_ms_per_document",
    "end_to_end_throughput_documents_per_second",
    "inference_cost_scope",
    "peak_sampled_rss_bytes",
    "artifact_size_bytes",
    "artifact_size_scope",
)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject invalid result files."""
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")

    return payload


def _require_mapping(value: Any, *, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object.")

    return value


def _validate_payload(
    payload: dict[str, Any],
    *,
    source: str,
    expected_split: str,
    expected_seed: int,
    expected_training_documents: int,
    expected_validation_documents: int,
    expected_class_count: int,
    require_test_data_excluded: bool,
) -> list[str]:
    """Validate shared dataset, split and evaluation conditions."""
    summary = payload.get("summary", {})

    if not isinstance(summary, dict):
        raise ValueError(f"{source} contains an invalid summary.")

    if payload.get("dataset") != "20newsgroups":
        raise ValueError(f"{source} does not identify the expected dataset.")

    if require_test_data_excluded and payload.get("test_data_used") is not False:
        raise ValueError(
            f"{source} does not explicitly confirm that test data was excluded."
        )

    if payload.get("development_split") != expected_split:
        raise ValueError(f"{source} uses a different development split.")

    if payload.get("random_seed") != expected_seed:
        raise ValueError(f"{source} uses an unexpected random seed.")

    if payload.get("primary_selection_metric") != "macro_f1":
        raise ValueError(
            f"{source} does not identify macro_f1 as its selection metric."
        )

    training_documents = payload.get(
        "training_documents",
        summary.get("training_documents"),
    )
    validation_documents = payload.get(
        "validation_documents",
        summary.get("validation_documents"),
    )

    if training_documents != expected_training_documents:
        raise ValueError(f"{source} has an unexpected training-document count.")

    if validation_documents != expected_validation_documents:
        raise ValueError(f"{source} has an unexpected validation-document count.")

    class_labels = payload.get("class_labels")

    if not isinstance(class_labels, list):
        raise ValueError(f"{source} does not contain class labels.")

    if len(class_labels) != expected_class_count:
        raise ValueError(f"{source} has an unexpected number of class labels.")

    if not all(isinstance(label, str) for label in class_labels):
        raise ValueError(f"{source} contains invalid class labels.")

    if len(set(class_labels)) != len(class_labels):
        raise ValueError(f"{source} contains duplicate class labels.")

    return class_labels


def _normalise_run(
    *,
    run: dict[str, Any],
    metrics: dict[str, Any],
    model_family: str,
    model: str,
    source: str,
    training_related_seconds: float,
    training_cost_scope: str,
    inference_seconds: float,
    inference_latency: float,
    inference_throughput: float | None,
    inference_cost_scope: str,
    artifact_size_bytes: int | None,
    artifact_size_scope: str,
    peak_sampled_rss_bytes: int | float | None = None,
) -> dict[str, Any]:
    """Convert an experiment record into a common comparison structure."""
    run_id = run.get("run_id")
    experiment_id = run.get("experiment_id")

    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{source} contains an invalid run ID.")

    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError(f"{source} contains an invalid experiment ID.")

    per_class = metrics.get("per_class")
    class_labels = metrics.get("class_labels")
    confusion_matrix = metrics.get("confusion_matrix")

    if not isinstance(per_class, dict):
        raise ValueError(f"{source} does not contain per-class metrics.")

    if not isinstance(class_labels, list):
        raise ValueError(f"{source} does not contain metric class labels.")

    if not isinstance(confusion_matrix, list):
        raise ValueError(f"{source} does not contain a confusion matrix.")

    if set(per_class) != set(class_labels):
        raise ValueError(f"{source} has inconsistent per-class metrics.")

    if len(confusion_matrix) != len(class_labels):
        raise ValueError(f"{source} has an invalid confusion matrix.")

    for row in confusion_matrix:
        if not isinstance(row, list) or len(row) != len(class_labels):
            raise ValueError(f"{source} has an invalid confusion matrix.")

    normalised_metrics: dict[str, float] = {}

    for metric_name in METRIC_NAMES:
        value = metrics.get(metric_name)

        if not isinstance(value, (int, float)):
            raise ValueError(f"{source} is missing {metric_name}.")

        score = float(value)

        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{source} contains an invalid {metric_name}.")

        normalised_metrics[metric_name] = score

    runtime_values = (
        training_related_seconds,
        inference_seconds,
        inference_latency,
    )

    if not all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in runtime_values
    ):
        raise ValueError(f"{source} contains invalid runtime measurements.")

    if min(runtime_values) < 0:
        raise ValueError(f"{source} contains negative runtime measurements.")

    if inference_throughput is not None:
        if not math.isfinite(inference_throughput) or inference_throughput < 0:
            raise ValueError(f"{source} contains invalid inference throughput.")

    if artifact_size_bytes is not None and artifact_size_bytes < 0:
        raise ValueError(f"{source} contains an invalid artifact size.")

    if peak_sampled_rss_bytes is not None:
        if peak_sampled_rss_bytes < 0:
            raise ValueError(f"{source} contains invalid memory measurements.")

    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "model_family": model_family,
        "model": model,
        "representation": run.get("representation"),
        "feature_type": run.get("feature_type"),
        "training_documents": int(run["training_documents"]),
        "validation_documents": int(run["validation_documents"]),
        **normalised_metrics,
        "training_related_seconds": float(training_related_seconds),
        "training_cost_scope": training_cost_scope,
        "end_to_end_inference_seconds": float(inference_seconds),
        "end_to_end_latency_ms_per_document": float(inference_latency),
        "end_to_end_throughput_documents_per_second": (
            float(inference_throughput) if inference_throughput is not None else None
        ),
        "inference_cost_scope": inference_cost_scope,
        "peak_sampled_rss_bytes": peak_sampled_rss_bytes,
        "artifact_size_bytes": artifact_size_bytes,
        "artifact_size_scope": artifact_size_scope,
        "class_labels": class_labels,
        "per_class_metrics": per_class,
        "confusion_matrix": confusion_matrix,
        "source_results_file": source,
    }


def _load_classical_runs(
    payload: dict[str, Any],
    *,
    source: str,
) -> list[dict[str, Any]]:
    runs = payload.get("runs")

    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{source} does not contain classical runs.")

    model_names = {
        "multinomial_nb": "Multinomial Naive Bayes",
        "logistic_regression": "Logistic Regression",
        "linear_svm": "Linear SVM",
    }

    results = []

    for run_value in runs:
        run = _require_mapping(run_value, description=source)
        classifier = run.get("classifier")

        if classifier not in model_names:
            raise ValueError(f"{source} contains an unknown classifier.")

        metrics = _require_mapping(
            run.get("metrics"),
            description=f"{source} metrics",
        )

        results.append(
            _normalise_run(
                run=run,
                metrics=metrics,
                model_family="Classical ML",
                model=model_names[classifier],
                source=source,
                training_related_seconds=run["fit_seconds"],
                training_cost_scope="Vectorizer and classifier fitting",
                inference_seconds=run["prediction_seconds"],
                inference_latency=run["latency_ms_per_document"],
                inference_throughput=run.get("throughput_documents_per_second"),
                inference_cost_scope=(
                    "Vectorizer transformation and classifier prediction"
                ),
                artifact_size_bytes=run.get("artifact_size_bytes"),
                artifact_size_scope="Fitted scikit-learn pipeline",
            )
        )

    return results


def _load_neural_run(
    payload: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    runs = payload.get("runs")

    if not isinstance(runs, list) or len(runs) != 1:
        raise ValueError(f"{source} must contain exactly one neural run.")

    run = _require_mapping(runs[0], description=source)
    metrics = _require_mapping(
        run.get("metrics"),
        description=f"{source} metrics",
    )

    model = run.get("model")

    if model not in {"TextCNN", "BiLSTM"}:
        raise ValueError(f"{source} contains an unsupported neural model.")

    return _normalise_run(
        run=run,
        metrics=metrics,
        model_family="Neural network",
        model=model,
        source=source,
        training_related_seconds=run["fit_seconds"],
        training_cost_scope=(
            "Neural training loop; vocabulary construction and encoding excluded"
        ),
        inference_seconds=run["prediction_seconds"],
        inference_latency=run["latency_ms_per_document"],
        inference_throughput=run.get("throughput_documents_per_second"),
        inference_cost_scope="Model prediction on encoded sequences",
        artifact_size_bytes=run.get("artifact_size_bytes"),
        artifact_size_scope="Neural model checkpoint and vocabulary",
    )


def _load_transformer_run(
    payload: dict[str, Any],
    *,
    source: str,
    model: str,
    frozen: bool,
) -> dict[str, Any]:
    summary = _require_mapping(
        payload.get("summary"),
        description=f"{source} summary",
    )
    metrics = _require_mapping(
        payload.get("metrics"),
        description=f"{source} metrics",
    )

    training_related_seconds = float(summary["fit_seconds"])

    if frozen:
        training_related_seconds += float(summary["training_embedding_seconds"])
        artifact_size = summary.get("classifier_artifact_size_bytes")
        artifact_scope = "Classifier only; pretrained encoder excluded"
        training_scope = (
            "Training embedding generation and classifier fitting; "
            "encoder loading excluded"
        )
        inference_scope = "Embedding generation and classifier prediction"
    else:
        artifact_size = summary.get("model_and_tokenizer_artifact_size_bytes")
        artifact_scope = "Fine-tuned model and tokenizer"
        training_scope = "Fine-tuning; model loading and initial tokenisation excluded"
        inference_scope = "Validation tokenisation and model prediction"

    run = {
        "run_id": summary["run_id"],
        "experiment_id": summary["experiment_id"],
        "training_documents": summary["training_documents"],
        "validation_documents": summary["validation_documents"],
        "representation": "message_body",
        "feature_type": None,
    }

    return _normalise_run(
        run=run,
        metrics=metrics,
        model_family="Transformer",
        model=model,
        source=source,
        training_related_seconds=training_related_seconds,
        training_cost_scope=training_scope,
        inference_seconds=summary["end_to_end_inference_seconds"],
        inference_latency=summary["end_to_end_latency_ms_per_document"],
        inference_throughput=summary.get("end_to_end_throughput_documents_per_second"),
        inference_cost_scope=inference_scope,
        artifact_size_bytes=artifact_size,
        artifact_size_scope=artifact_scope,
        peak_sampled_rss_bytes=summary.get("peak_sampled_rss_bytes"),
    )


def _best_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the highest validation macro F1 with deterministic tie-breaking."""
    if not runs:
        raise ValueError("Cannot select a best run from an empty collection.")

    return sorted(
        runs,
        key=lambda run: (-run["macro_f1"], run["run_id"]),
    )[0]


def _best_classical_by_classifier(
    runs: list[dict[str, Any]],
    *,
    representation: str | None = None,
) -> list[dict[str, Any]]:
    model_names = (
        "Multinomial Naive Bayes",
        "Logistic Regression",
        "Linear SVM",
    )

    results = []

    for model_name in model_names:
        candidates = [
            run
            for run in runs
            if run["model"] == model_name
            and (representation is None or run["representation"] == representation)
        ]

        if candidates:
            results.append(_best_run(candidates))

    return results


def _build_family_summaries(
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []

    for family in ("Classical ML", "Neural network", "Transformer"):
        family_runs = [run for run in runs if run["model_family"] == family]

        if not family_runs:
            continue

        best = _best_run(family_runs)

        summaries.append(
            {
                "model_family": family,
                "best_model": best["model"],
                "best_run_id": best["run_id"],
                "macro_f1": best["macro_f1"],
                "accuracy": best["accuracy"],
                "weighted_f1": best["weighted_f1"],
                "training_related_seconds": (best["training_related_seconds"]),
                "end_to_end_inference_seconds": (best["end_to_end_inference_seconds"]),
            }
        )

    return summaries


def build_model_comparison(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Load, validate and consolidate saved development experiment results."""
    data_config = config["data"]
    input_config = config["inputs"]
    comparison_config = config["comparison"]

    if comparison_config.get("primary_metric") != "macro_f1":
        raise ValueError("The comparison primary metric must be macro_f1.")

    expected_split = str(data_config["development_split"])
    expected_seed = int(config["project"]["random_seed"])
    expected_training = int(data_config["expected_training_documents"])
    expected_validation = int(data_config["expected_validation_documents"])
    expected_classes = int(data_config["expected_class_count"])

    input_paths = {
        name: project_root / relative_path
        for name, relative_path in input_config.items()
    }

    payloads = {name: _read_json(path) for name, path in input_paths.items()}

    class_labels: list[str] | None = None

    for name, payload in payloads.items():
        current_labels = _validate_payload(
            payload,
            source=name,
            expected_split=expected_split,
            expected_seed=expected_seed,
            expected_training_documents=expected_training,
            expected_validation_documents=expected_validation,
            expected_class_count=expected_classes,
            require_test_data_excluded=bool(
                comparison_config["require_test_data_excluded"]
            ),
        )

        if class_labels is None:
            class_labels = current_labels
        elif current_labels != class_labels:
            raise ValueError(f"{name} uses a different class-label ordering.")

    classical = _load_classical_runs(
        payloads["classical_results"],
        source="classical_results",
    )

    neural = [
        _load_neural_run(
            payloads["textcnn_results"],
            source="textcnn_results",
        ),
        _load_neural_run(
            payloads["bilstm_results"],
            source="bilstm_results",
        ),
    ]

    transformers = [
        _load_transformer_run(
            payloads["distilbert_results"],
            source="distilbert_results",
            model="DistilBERT",
            frozen=False,
        ),
        _load_transformer_run(
            payloads["frozen_embeddings_results"],
            source="frozen_embeddings_results",
            model="Frozen SentenceTransformer",
            frozen=True,
        ),
    ]

    runs = classical + neural + transformers

    expected_classical_runs = int(comparison_config["expected_classical_run_count"])
    expected_total_runs = int(comparison_config["expected_total_runs"])

    if len(classical) != expected_classical_runs:
        raise ValueError(
            f"Expected {expected_classical_runs} classical runs, "
            f"but found {len(classical)}."
        )

    if len(runs) != expected_total_runs:
        raise ValueError(
            f"Expected {expected_total_runs} total runs, but found {len(runs)}."
        )

    run_ids = [run["run_id"] for run in runs]

    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Comparison results contain duplicate run IDs.")

    for run in runs:
        if run["training_documents"] != expected_training:
            raise ValueError(
                f"{run['run_id']} has an unexpected training-document count."
            )

        if run["validation_documents"] != expected_validation:
            raise ValueError(
                f"{run['run_id']} has an unexpected validation-document count."
            )

        if run["class_labels"] != class_labels:
            raise ValueError(f"{run['run_id']} has inconsistent class labels.")

    ranked_runs = sorted(
        runs,
        key=lambda run: (-run["macro_f1"], run["run_id"]),
    )

    for rank, run in enumerate(ranked_runs, start=1):
        run["rank"] = rank

    best_classical = _best_classical_by_classifier(classical)

    best_classical_by_representation = []

    for representation in ("full_message", "message_body"):
        candidates = [
            run for run in classical if run["representation"] == representation
        ]

        if candidates:
            best_classical_by_representation.append(_best_run(candidates))

    best_classical_by_classifier_and_representation = _best_classical_by_classifier(
        classical,
        representation="full_message",
    ) + _best_classical_by_classifier(
        classical,
        representation="message_body",
    )

    matched_message_body_comparison = (
        _best_classical_by_classifier(
            classical,
            representation="message_body",
        )
        + neural
        + transformers
    )

    matched_message_body_comparison = sorted(
        matched_message_body_comparison,
        key=lambda run: (-run["macro_f1"], run["run_id"]),
    )

    for rank, run in enumerate(matched_message_body_comparison, start=1):
        run["matched_representation_rank"] = rank

    return {
        "metadata": {
            "dataset": data_config["dataset"],
            "development_split": expected_split,
            "random_seed": expected_seed,
            "training_documents": expected_training,
            "validation_documents": expected_validation,
            "class_count": expected_classes,
            "primary_selection_metric": "macro_f1",
            "test_data_used": False,
            "experiment_count": len(ranked_runs),
        },
        "runs": ranked_runs,
        "best_classical_by_classifier": best_classical,
        "best_classical_by_representation": (best_classical_by_representation),
        "best_classical_by_classifier_and_representation": (
            best_classical_by_classifier_and_representation
        ),
        "matched_message_body_comparison": matched_message_body_comparison,
        "best_by_family": _build_family_summaries(ranked_runs),
        "best_overall_validation": ranked_runs[0],
    }


def write_model_comparison(
    comparison: dict[str, Any],
    *,
    config: dict[str, Any],
    project_root: Path,
) -> tuple[Path, Path]:
    """Write consolidated comparison results to CSV and JSON."""
    output_config = config["outputs"]

    csv_path = project_root / output_config["comparison_metrics_csv"]
    json_path = project_root / output_config["comparison_results_json"]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for run in comparison["runs"]:
            writer.writerow({field: run.get(field) for field in CSV_FIELDS})

    json_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return csv_path, json_path
