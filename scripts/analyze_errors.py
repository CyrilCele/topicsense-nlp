"""Run validation error analysis using saved model predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from topicsense.evaluation.error_analysis import (
    build_error_analysis,
    write_error_analysis,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")

    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON Lines records and validate their structure."""

    if not path.is_file():
        raise FileNotFoundError(path)

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} in {path}"
                )

            records.append(record)

    return records


def main() -> None:
    """Build error-analysis reports from previously saved predictions."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyse validation errors using saved predictions. "
            "This command does not train or evaluate a model."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root directory.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments.yaml"),
        help="Path to the experiment configuration.",
    )

    args = parser.parse_args()

    project_root = args.project_root.resolve()
    config_path = args.config

    if not config_path.is_absolute():
        config_path = project_root / config_path

    config = _load_yaml(config_path)

    data_config = config["data"]
    error_config = config["error_analysis"]
    selected_run = error_config["selected_run"]
    configured_outputs = error_config["outputs"]

    required_outputs = {
        "predictions_jsonl",
        "summary_json",
        "per_class_csv",
        "confusion_pairs_csv",
        "examples_csv",
    }

    missing_outputs = required_outputs - set(configured_outputs)

    if missing_outputs:
        raise ValueError(
            "Missing error-analysis output configuration entries: "
            f"{sorted(missing_outputs)}"
        )

    predictions_path = project_root / configured_outputs["predictions_jsonl"]

    comparison_path = project_root / config["outputs"]["comparison_results_json"]

    raw_dataset_path = project_root / error_config["classical_config"]

    classical_config = _load_yaml(raw_dataset_path)

    raw_data_path = project_root / classical_config["data"]["raw_dataset_path"]

    split_path = project_root / data_config["development_split"]

    # Verify the saved comparison before processing any examples.
    if not comparison_path.is_file():
        raise FileNotFoundError(
            "Run the model comparison before error analysis. "
            f"Missing comparison results: {comparison_path}"
        )

    with comparison_path.open("r", encoding="utf-8") as file:
        comparison = json.load(file)

    if not isinstance(comparison, dict):
        raise ValueError("The saved model comparison must be a JSON object.")

    comparison_metadata = comparison.get("metadata", {})
    best_run = comparison.get("best_overall_validation", {})

    if comparison_metadata.get("test_data_used") is not False:
        raise ValueError("The model comparison does not confirm test-data exclusion.")

    if best_run.get("run_id") != ("linear_svm__tfidf__full_message"):
        raise ValueError(
            "The saved best overall validation run is not the "
            "expected Linear SVM configuration."
        )

    selected_run_id = (
        f"{selected_run['classifier']}__"
        f"{selected_run['feature_type']}__"
        f"{selected_run['representation']}"
    )

    if best_run.get("run_id") != selected_run_id:
        raise ValueError(
            "The selected error-analysis configuration does not match "
            "the best overall validation run."
        )

    expected_split = data_config["development_split"]

    if comparison_metadata.get("development_split") != expected_split:
        raise ValueError("The comparison and error-analysis development splits differ.")

    # Reuse existing predictions. No model fitting takes place here.
    predictions = _load_jsonl(predictions_path)
    source_records = _load_jsonl(raw_data_path)

    class_labels = best_run.get("class_labels")

    if not isinstance(class_labels, list) or not all(
        isinstance(label, str) for label in class_labels
    ):
        raise ValueError("The saved best run does not contain valid class labels.")

    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    with split_path.open("r", encoding="utf-8") as file:
        split_manifest = json.load(file)

    validation_ids = split_manifest.get("validation_record_ids")

    if not isinstance(validation_ids, list):
        raise ValueError(
            "The development split does not contain validation record IDs."
        )

    prediction_ids = {prediction.get("record_id") for prediction in predictions}

    if prediction_ids != set(validation_ids):
        raise ValueError("Saved predictions do not match the fixed validation split.")

    if len(predictions) != len(validation_ids):
        raise ValueError("The saved predictions contain duplicate or missing records.")

    if len(predictions) != data_config["expected_validation_documents"]:
        raise ValueError(
            "The saved prediction count does not match the configured "
            "validation document count."
        )

    analysis = build_error_analysis(
        predictions,
        source_records,
        class_labels=class_labels,
        selected_run_id=selected_run_id,
        dataset=data_config["dataset"],
        development_split=expected_split,
        max_examples_per_pair=error_config["max_examples_per_confusion_pair"],
    )

    output_paths = write_error_analysis(
        analysis,
        outputs=configured_outputs,
        project_root=project_root,
    )

    print("Validation error analysis completed.")
    print(f"Selected configuration: {selected_run_id}")
    print(f"Validation documents: {analysis['metadata']['validation_documents']}")
    print(f"Misclassified documents: {analysis['metadata']['misclassified_documents']}")
    print(f"Validation macro F1: {analysis['metrics']['macro_f1']:.4f}")

    print("\nGenerated outputs:")

    for name, path in output_paths.items():
        print(f"{name}: {path}")

    print("\nNo model training was performed.")
    print("Official test data was not used.")


if __name__ == "__main__":
    main()
