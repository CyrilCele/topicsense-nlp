"""Command-line entry point for consolidated model evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from topicsense.evaluation.comparison import (
    build_model_comparison,
    write_model_comparison,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Consolidate and compare TopicSense-NLP model results."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments.yaml"),
        help="Path to the model comparison configuration.",
    )

    return parser.parse_args()


def main() -> None:
    """Load saved results and generate the consolidated comparison."""
    arguments = parse_arguments()
    project_root = Path.cwd().resolve()

    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config: Any = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The comparison configuration must be a YAML mapping.")

    comparison = build_model_comparison(
        config,
        project_root=project_root,
    )

    metrics_path, results_path = write_model_comparison(
        comparison,
        config=config,
        project_root=project_root,
    )

    print("\nModel comparison completed.")
    print(f"Experiments compared: {comparison['metadata']['experiment_count']}")
    print("\nBest validation result:")
    print(comparison["best_overall_validation"]["model"])
    print(
        "Macro F1:",
        f"{comparison['best_overall_validation']['macro_f1']:.4f}",
    )
    print(f"\nMetrics CSV: {metrics_path}")
    print(f"Detailed results: {results_path}")


if __name__ == "__main__":
    main()
