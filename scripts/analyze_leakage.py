"""Command-line entry point for duplicate and leakage analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from topicsense.data.body_audit import run_body_audit
from topicsense.data.leakage import analyze_dataset_leakage

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(value).expanduser()

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_configuration(config_path: Path) -> dict[str, Any]:
    """Load the dataset configuration."""
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
    """Execute duplicate detection and leakage analysis."""
    parser = argparse.ArgumentParser(
        description="Analyse duplicate documents and metadata leakage."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data.yaml",
        help="Path to the dataset YAML configuration.",
    )

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Minimum Jaccard similarity for reporting near duplicates.",
    )

    parser.add_argument(
        "--shingle-size",
        type=int,
        default=5,
        help="Number of consecutive words in each shingle.",
    )

    parser.add_argument(
        "--num-permutations",
        type=int,
        default=32,
        help="Number of MinHash permutations.",
    )

    parser.add_argument(
        "--num-bands",
        type=int,
        default=8,
        help="Number of LSH bands.",
    )

    args = parser.parse_args()

    if args.shingle_size < 1:
        parser.error("--shingle-size must be at least 1.")

    if args.num_permutations < 1 or args.num_bands < 1:
        parser.error("MinHash permutations and LSH bands must be positive.")

    if args.num_permutations % args.num_bands != 0:
        parser.error("--num-permutations must be divisible by --num-bands.")

    if not 0.0 < args.similarity_threshold <= 1.0:
        parser.error("--similarity-threshold must be in (0, 1].")

    config_path = args.config

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    configuration = load_configuration(config_path)
    dataset_config = configuration["dataset"]

    if dataset_config.get("name") != "20newsgroups":
        raise ValueError("Unsupported dataset.")

    dataset_path = resolve_project_path(dataset_config["output_file"])

    random_seed = int(configuration["project"]["random_seed"])

    report = analyze_dataset_leakage(
        dataset_path=dataset_path,
        shingle_size=args.shingle_size,
        num_permutations=args.num_permutations,
        num_bands=args.num_bands,
        similarity_threshold=args.similarity_threshold,
        random_seed=random_seed,
    )

    report_path = PROJECT_ROOT / "results" / "logs" / "dataset_leakage_analysis.json"

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)
        file.write("\n")

    print("\nFirst-stage duplicate and leakage analysis completed.")
    print(json.dumps(report, indent=4, ensure_ascii=False))
    print(f"\nReport saved to: {report_path}")

    body_report = run_body_audit(
        input_path="data/raw/20newsgroups/20newsgroups_raw.jsonl",
        output_path="results/logs/dataset_body_duplicate_audit.json",
    )

    print("\nSecond-stage document-body audit")
    print(
        json.dumps(
            body_report["analysis"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
