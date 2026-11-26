"""Tests for validation error analysis and text-free reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from topicsense.evaluation.error_analysis import (
    build_error_analysis,
    write_error_analysis,
)


@pytest.fixture
def class_labels() -> list[str]:
    """Return a small set of classification labels."""

    return ["computer", "religion", "science"]


@pytest.fixture
def source_records() -> list[dict[str, Any]]:
    """Return synthetic source records from the training partition."""

    return [
        {
            "record_id": "doc-001",
            "source_split": "train",
            "label": "computer",
            "text": (
                "Subject: Computer hardware\n\n"
                "The graphics driver causes display problems."
            ),
        },
        {
            "record_id": "doc-002",
            "source_split": "train",
            "label": "computer",
            "text": (
                "Subject: Graphics\n\nThe graphics card supports a new display mode."
            ),
        },
        {
            "record_id": "doc-003",
            "source_split": "train",
            "label": "religion",
            "text": (
                "Subject: Religious discussion\n\n"
                "This message discusses religious beliefs."
            ),
        },
        {
            "record_id": "doc-004",
            "source_split": "train",
            "label": "religion",
            "text": (
                "Subject: Religious history\n\n"
                "The historical discussion concerns religious traditions."
            ),
        },
        {
            "record_id": "doc-005",
            "source_split": "train",
            "label": "science",
            "text": (
                "Subject: Astronomy\n\nThe telescope has a useful new imaging system."
            ),
        },
        {
            "record_id": "doc-006",
            "source_split": "train",
            "label": "science",
            "text": (
                "Subject: Space research\n\nThe research examines distant planets."
            ),
        },
        {
            "record_id": "test-001",
            "source_split": "test",
            "label": "computer",
            "text": "This is an official test record.",
        },
    ]


@pytest.fixture
def predictions() -> list[dict[str, str]]:
    """Return synthetic validation predictions."""

    return [
        {
            "record_id": "doc-001",
            "actual_label": "computer",
            "predicted_label": "computer",
        },
        {
            "record_id": "doc-002",
            "actual_label": "computer",
            "predicted_label": "religion",
        },
        {
            "record_id": "doc-003",
            "actual_label": "religion",
            "predicted_label": "religion",
        },
        {
            "record_id": "doc-004",
            "actual_label": "religion",
            "predicted_label": "computer",
        },
        {
            "record_id": "doc-005",
            "actual_label": "science",
            "predicted_label": "science",
        },
        {
            "record_id": "doc-006",
            "actual_label": "science",
            "predicted_label": "science",
        },
    ]


def test_build_error_analysis_calculates_class_errors(
    predictions: list[dict[str, str]],
    source_records: list[dict[str, Any]],
    class_labels: list[str],
) -> None:
    """Class statistics and directional confusion pairs are calculated."""

    analysis = build_error_analysis(
        predictions,
        source_records,
        class_labels=class_labels,
        selected_run_id="linear_svm__tfidf__full_message",
        dataset="20newsgroups",
        development_split="official_training_split.json",
    )

    assert analysis["metadata"]["validation_documents"] == 6
    assert analysis["metadata"]["misclassified_documents"] == 2
    assert analysis["metadata"]["correctly_classified_documents"] == 4
    assert analysis["metadata"]["class_count"] == 3
    assert analysis["metadata"]["test_data_used"] is False

    assert analysis["metadata"]["example_text_representation"] == (
        "Source message text excluded from generated reports. "
        "Publication excerpts require manual review and redaction."
    )

    assert analysis["metrics"]["accuracy"] == pytest.approx(4 / 6)

    per_class = {item["class_label"]: item for item in analysis["per_class"]}

    assert per_class["computer"]["misclassified_documents"] == 1
    assert per_class["religion"]["misclassified_documents"] == 1
    assert per_class["science"]["misclassified_documents"] == 0

    confusion_pairs = {
        (
            item["actual_label"],
            item["predicted_label"],
        ): item["document_count"]
        for item in analysis["confusion_pairs"]
    }

    assert confusion_pairs[("computer", "religion")] == 1
    assert confusion_pairs[("religion", "computer")] == 1

    assert len(analysis["misclassified_examples"]) == 2

    for example in analysis["misclassified_examples"]:
        assert "text" not in example
        assert "record_id" in example
        assert "actual_label" in example
        assert "predicted_label" in example
        assert example["excerpt_review_status"] == "withheld_pending_manual_review"


def test_build_error_analysis_excludes_source_message_text(
    predictions: list[dict[str, str]],
    source_records: list[dict[str, Any]],
    class_labels: list[str],
) -> None:
    """Original message content is not included in generated analysis."""

    analysis = build_error_analysis(
        predictions,
        source_records,
        class_labels=class_labels,
        selected_run_id="linear_svm__tfidf__full_message",
        dataset="20newsgroups",
        development_split="official_training_split.json",
    )

    analysis_text = json.dumps(analysis, ensure_ascii=False)

    sensitive_phrases = [
        "graphics card supports a new display mode",
        "historical discussion concerns religious traditions",
        "official test record",
        "Subject: Graphics",
    ]

    for phrase in sensitive_phrases:
        assert phrase not in analysis_text


def test_build_error_analysis_rejects_official_test_prediction(
    predictions: list[dict[str, str]],
    source_records: list[dict[str, Any]],
    class_labels: list[str],
) -> None:
    """Predictions cannot refer to official test records."""

    invalid_predictions = [
        *predictions,
        {
            "record_id": "test-001",
            "actual_label": "computer",
            "predicted_label": "computer",
        },
    ]

    with pytest.raises(ValueError, match="official training partition"):
        build_error_analysis(
            invalid_predictions,
            source_records,
            class_labels=class_labels,
            selected_run_id="linear_svm__tfidf__full_message",
            dataset="20newsgroups",
            development_split="official_training_split.json",
        )


def test_build_error_analysis_rejects_duplicate_predictions(
    predictions: list[dict[str, str]],
    source_records: list[dict[str, Any]],
    class_labels: list[str],
) -> None:
    """Duplicate prediction record IDs are rejected."""

    invalid_predictions = [
        *predictions,
        predictions[0],
    ]

    with pytest.raises(ValueError, match="Duplicate prediction ID"):
        build_error_analysis(
            invalid_predictions,
            source_records,
            class_labels=class_labels,
            selected_run_id="linear_svm__tfidf__full_message",
            dataset="20newsgroups",
            development_split="official_training_split.json",
        )


def test_build_error_analysis_rejects_actual_label_mismatch(
    predictions: list[dict[str, str]],
    source_records: list[dict[str, Any]],
    class_labels: list[str],
) -> None:
    """Prediction labels must agree with the source records."""

    invalid_predictions = [
        *predictions,
    ]

    invalid_predictions[0] = {
        **invalid_predictions[0],
        "actual_label": "religion",
    }

    with pytest.raises(ValueError, match="Actual label mismatch"):
        build_error_analysis(
            invalid_predictions,
            source_records,
            class_labels=class_labels,
            selected_run_id="linear_svm__tfidf__full_message",
            dataset="20newsgroups",
            development_split="official_training_split.json",
        )


def test_build_error_analysis_rejects_unknown_predicted_label(
    predictions: list[dict[str, str]],
    source_records: list[dict[str, Any]],
    class_labels: list[str],
) -> None:
    """Predictions containing unknown class labels are rejected."""

    invalid_predictions = [
        *predictions,
    ]

    invalid_predictions[0] = {
        **invalid_predictions[0],
        "predicted_label": "unknown",
    }

    with pytest.raises(ValueError, match="Unknown predicted label"):
        build_error_analysis(
            invalid_predictions,
            source_records,
            class_labels=class_labels,
            selected_run_id="linear_svm__tfidf__full_message",
            dataset="20newsgroups",
            development_split="official_training_split.json",
        )


def test_build_error_analysis_rejects_duplicate_source_records(
    predictions: list[dict[str, str]],
    source_records: list[dict[str, Any]],
    class_labels: list[str],
) -> None:
    """Duplicate source record IDs in the training partition are rejected."""

    invalid_records = [
        *source_records,
        source_records[0],
    ]

    with pytest.raises(ValueError, match="Duplicate source record ID"):
        build_error_analysis(
            predictions,
            invalid_records,
            class_labels=class_labels,
            selected_run_id="linear_svm__tfidf__full_message",
            dataset="20newsgroups",
            development_split="official_training_split.json",
        )


def test_build_error_analysis_rejects_empty_predictions(
    source_records: list[dict[str, Any]],
    class_labels: list[str],
) -> None:
    """An empty prediction collection is rejected."""

    with pytest.raises(ValueError, match="Predictions cannot be empty"):
        build_error_analysis(
            [],
            source_records,
            class_labels=class_labels,
            selected_run_id="linear_svm__tfidf__full_message",
            dataset="20newsgroups",
            development_split="official_training_split.json",
        )


def test_build_error_analysis_rejects_invalid_example_limit(
    predictions: list[dict[str, str]],
    source_records: list[dict[str, Any]],
    class_labels: list[str],
) -> None:
    """The maximum number of examples per confusion pair must be positive."""

    with pytest.raises(ValueError, match="max_examples_per_pair"):
        build_error_analysis(
            predictions,
            source_records,
            class_labels=class_labels,
            selected_run_id="linear_svm__tfidf__full_message",
            dataset="20newsgroups",
            development_split="official_training_split.json",
            max_examples_per_pair=0,
        )


def test_write_error_analysis_creates_all_outputs(
    tmp_path: Path,
    predictions: list[dict[str, str]],
    source_records: list[dict[str, Any]],
    class_labels: list[str],
) -> None:
    """JSON and CSV reports are written without source message text."""

    analysis = build_error_analysis(
        predictions,
        source_records,
        class_labels=class_labels,
        selected_run_id="linear_svm__tfidf__full_message",
        dataset="20newsgroups",
        development_split="official_training_split.json",
    )

    outputs = {
        "predictions_jsonl": "results/logs/error_analysis_predictions.jsonl",
        "summary_json": "results/logs/error_analysis.json",
        "per_class_csv": "results/tables/error_analysis_per_class.csv",
        "confusion_pairs_csv": ("results/tables/error_analysis_confusion_pairs.csv"),
        "examples_csv": ("results/tables/error_analysis_misclassified_examples.csv"),
    }

    written_paths = write_error_analysis(
        analysis,
        outputs=outputs,
        project_root=tmp_path,
    )

    assert set(written_paths) == set(outputs)

    summary_path = tmp_path / outputs["summary_json"]
    per_class_path = tmp_path / outputs["per_class_csv"]
    confusion_path = tmp_path / outputs["confusion_pairs_csv"]
    examples_path = tmp_path / outputs["examples_csv"]

    assert summary_path.is_file()
    assert per_class_path.is_file()
    assert confusion_path.is_file()
    assert examples_path.is_file()

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert saved_summary["metadata"]["test_data_used"] is False
    assert len(saved_summary["misclassified_examples"]) == 2

    for example in saved_summary["misclassified_examples"]:
        assert "text" not in example

    with per_class_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        per_class_rows = list(csv.DictReader(file))

    assert len(per_class_rows) == len(class_labels)

    with confusion_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        confusion_rows = list(csv.DictReader(file))

    assert len(confusion_rows) == 2

    with examples_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        example_rows = list(csv.DictReader(file))

    assert len(example_rows) == 2

    assert set(example_rows[0]) == {
        "record_id",
        "actual_label",
        "predicted_label",
        "excerpt_review_status",
    }

    exported_text = summary_path.read_text(encoding="utf-8") + examples_path.read_text(
        encoding="utf-8"
    )

    for record in source_records:
        if record["source_split"] == "train":
            assert record["text"] not in exported_text
