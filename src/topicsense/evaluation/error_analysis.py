"""Validation error analysis for supervised text classification."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from topicsense.evaluation.metrics import calculate_classification_metrics


def build_error_analysis(
    predictions: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    class_labels: list[str],
    selected_run_id: str,
    dataset: str,
    development_split: str,
    max_examples_per_pair: int = 3,
) -> dict[str, Any]:
    """Build class-level statistics and text-free confusion reports."""

    if not predictions:
        raise ValueError("Predictions cannot be empty.")

    if not class_labels or len(class_labels) != len(set(class_labels)):
        raise ValueError("Class labels must be non-empty and unique.")

    if max_examples_per_pair < 1:
        raise ValueError("max_examples_per_pair must be at least 1.")

    records_by_id: dict[str, dict[str, Any]] = {}

    for record in records:
        record_id = record.get("record_id")

        if not isinstance(record_id, str) or not record_id:
            raise ValueError("Every source record must have a valid record_id.")

        if record.get("source_split") != "train":
            continue

        if record_id in records_by_id:
            raise ValueError(f"Duplicate source record ID: {record_id}")

        records_by_id[record_id] = record

    prediction_ids: set[str] = set()
    y_true: list[str] = []
    y_pred: list[str] = []
    misclassified: list[dict[str, Any]] = []

    valid_labels = set(class_labels)

    for prediction in predictions:
        record_id = prediction.get("record_id")
        actual_label = prediction.get("actual_label")
        predicted_label = prediction.get("predicted_label")

        if not isinstance(record_id, str) or not record_id:
            raise ValueError("Every prediction must have a valid record_id.")

        if record_id in prediction_ids:
            raise ValueError(f"Duplicate prediction ID: {record_id}")

        prediction_ids.add(record_id)

        if record_id not in records_by_id:
            raise ValueError(
                f"Prediction record {record_id} is not in the "
                "official training partition."
            )

        source_record = records_by_id[record_id]

        if actual_label != source_record.get("label"):
            raise ValueError(f"Actual label mismatch for record {record_id}.")

        if not isinstance(actual_label, str) or actual_label not in valid_labels:
            raise ValueError(f"Unknown actual label: {actual_label}")

        if not isinstance(predicted_label, str) or predicted_label not in valid_labels:
            raise ValueError(f"Unknown predicted label: {predicted_label}")

        y_true.append(actual_label)
        y_pred.append(predicted_label)

        if actual_label != predicted_label:
            misclassified.append(
                {
                    "record_id": record_id,
                    "actual_label": actual_label,
                    "predicted_label": predicted_label,
                    "excerpt_review_status": "withheld_pending_manual_review",
                }
            )

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        class_labels=class_labels,
    )

    per_class: list[dict[str, Any]] = []

    for class_index, label in enumerate(class_labels):
        class_metrics = metrics["per_class"][label]
        matrix = metrics["confusion_matrix"]

        correct = matrix[class_index][class_index]
        support = class_metrics["support"]

        false_positives = sum(
            matrix[row][class_index]
            for row in range(len(class_labels))
            if row != class_index
        )

        per_class.append(
            {
                "class_label": label,
                "support": support,
                "correct_predictions": correct,
                "misclassified_documents": support - correct,
                "false_positives": false_positives,
                "precision": class_metrics["precision"],
                "recall": class_metrics["recall"],
                "f1_score": class_metrics["f1_score"],
            }
        )

    confusion_pairs: list[dict[str, Any]] = []

    for actual_index, actual_label in enumerate(class_labels):
        for predicted_index, predicted_label in enumerate(class_labels):
            if actual_index == predicted_index:
                continue

            count = metrics["confusion_matrix"][actual_index][predicted_index]

            if count > 0:
                confusion_pairs.append(
                    {
                        "actual_label": actual_label,
                        "predicted_label": predicted_label,
                        "document_count": count,
                    }
                )

    confusion_pairs.sort(
        key=lambda item: (
            -item["document_count"],
            item["actual_label"],
            item["predicted_label"],
        )
    )

    examples_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for example in sorted(misclassified, key=lambda item: item["record_id"]):
        label_pair = (
            example["actual_label"],
            example["predicted_label"],
        )

        pair_examples = examples_by_pair.setdefault(label_pair, [])

        if len(pair_examples) < max_examples_per_pair:
            pair_examples.append(example)

    misclassified_examples: list[dict[str, Any]] = []

    for confusion_pair in confusion_pairs:
        label_pair = (
            confusion_pair["actual_label"],
            confusion_pair["predicted_label"],
        )

        misclassified_examples.extend(examples_by_pair.get(label_pair, []))

    total_documents = len(predictions)
    misclassified_count = len(misclassified)

    return {
        "metadata": {
            "dataset": dataset,
            "development_split": development_split,
            "selected_run_id": selected_run_id,
            "validation_documents": total_documents,
            "misclassified_documents": misclassified_count,
            "correctly_classified_documents": (total_documents - misclassified_count),
            "class_count": len(class_labels),
            "test_data_used": False,
            "example_text_representation": (
                "Source message text excluded from generated reports. "
                "Publication excerpts require manual review and redaction."
            ),
            "max_examples_per_confusion_pair": max_examples_per_pair,
        },
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key
            not in {
                "per_class",
                "confusion_matrix",
                "class_labels",
            }
        },
        "per_class": per_class,
        "confusion_pairs": confusion_pairs,
        "misclassified_examples": misclassified_examples,
    }


def write_error_analysis(
    analysis: dict[str, Any],
    *,
    outputs: dict[str, str],
    project_root: Path,
) -> dict[str, str]:
    """Write text-free error-analysis JSON and CSV reports."""

    required_outputs = {
        "summary_json",
        "per_class_csv",
        "confusion_pairs_csv",
        "examples_csv",
    }

    missing_outputs = required_outputs.difference(outputs)

    if missing_outputs:
        missing = ", ".join(sorted(missing_outputs))
        raise ValueError(f"Missing required output paths: {missing}")

    output_paths = {key: project_root / value for key, value in outputs.items()}

    for output_path in output_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path = output_paths["summary_json"]

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            analysis,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    csv_rows_by_output: dict[str, list[dict[str, Any]]] = {
        "per_class_csv": analysis["per_class"],
        "confusion_pairs_csv": analysis["confusion_pairs"],
        "examples_csv": analysis["misclassified_examples"],
    }

    for output_key, csv_rows in csv_rows_by_output.items():
        output_path = output_paths[output_key]

        if not csv_rows:
            output_path.write_text("", encoding="utf-8")
            continue

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(csv_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(csv_rows)

    return {key: str(path) for key, path in output_paths.items()}
