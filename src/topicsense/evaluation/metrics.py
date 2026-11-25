"""Classification metrics and class-level evaluation."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    *,
    class_labels: list[str],
) -> dict[str, Any]:
    """Calculate aggregate, per-class and confusion-matrix results."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths.")

    if not y_true:
        raise ValueError("Evaluation data cannot be empty.")

    report = classification_report(
        y_true,
        y_pred,
        labels=class_labels,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )

    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1_score": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in class_labels
    }

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=class_labels,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=class_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "class_labels": class_labels,
    }
