"""Command-line entry point for dataset validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from topicsense.data.validation import validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(value).expanduser()

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_configuration(config_path: Path) -> dict[str, Any]:
    """Load and validate the dataset configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    if not isinstance(configuration, dict):
        raise ValueError("The configuration must contain a YAML mapping.")

    if "dataset" not in configuration:
        raise ValueError("Missing required 'dataset' configuration.")

    return configuration


def main() -> None:
    """Execute the dataset validation workflow."""
    parser = argparse.ArgumentParser(
        description="Validate the acquired 20 Newsgroups dataset."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data.yaml",
        help="Path to the dataset YAML configuration.",
    )

    args = parser.parse_args()

    config_path = args.config

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    configuration = load_configuration(config_path)
    dataset_config = configuration["dataset"]

    if dataset_config.get("name") != "20newsgroups":
        raise ValueError("Unsupported dataset.")

    dataset_path = resolve_project_path(dataset_config["output_file"])

    manifest_path = resolve_project_path(dataset_config["manifest_file"])

    expected_documents = int(dataset_config["expected_documents"])

    expected_classes = int(dataset_config["expected_classes"])

    report = validate_dataset(
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        expected_documents=expected_documents,
        expected_classes=expected_classes,
    )

    report_path = PROJECT_ROOT / "results" / "logs" / "dataset_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)
        file.write("\n")

    print("\nDataset validation report:")
    print(json.dumps(report, indent=4, ensure_ascii=False))
    print(f"\nValidation report saved to: {report_path}")

    if report["validation_status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
