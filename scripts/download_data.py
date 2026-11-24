from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from topicsense.data.ingestion import download_20newsgroups

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

    with config_path.open("r", encoding="utf-8") as config_file:
        configuration = yaml.safe_load(config_file)

    if not isinstance(configuration, dict):
        raise ValueError("The configuration must contain a YAML mapping.")

    if "dataset" not in configuration:
        raise ValueError("Missing required 'dataset' configuration.")

    return configuration


def main() -> None:
    """Execute the dataset acquisition workflow."""
    parser = argparse.ArgumentParser(
        description="Download and preserve the 20 Newsgroups dataset."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data.yaml",
        help="Path to the dataset YAML configuration.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing raw dataset and manifest files.",
    )

    args = parser.parse_args()

    config_path = args.config

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    configuration = load_configuration(config_path)
    dataset_config = configuration["dataset"]

    if dataset_config.get("name") != "20newsgroups":
        raise ValueError(
            "This acquisition script currently supports only the 20 Newsgroups dataset."
        )

    manifest = download_20newsgroups(
        data_home=resolve_project_path(dataset_config["cache_dir"]),
        output_path=resolve_project_path(dataset_config["output_file"]),
        manifest_path=resolve_project_path(dataset_config["manifest_file"]),
        subsets=tuple(dataset_config["subsets"]),
        shuffle=bool(dataset_config["shuffle"]),
        overwrite=args.overwrite,
    )

    print("\nDataset acquisition completed.")
    print(json.dumps(manifest, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
