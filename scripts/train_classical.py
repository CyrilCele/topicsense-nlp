"""Command-line entry point for classical baseline experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from topicsense.training.classical import run_classical_experiments


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate classical TopicSense-NLP baselines."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/classical_ml.yaml"),
        help="Path to the classical ML YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Load configuration and execute the baseline experiment matrix."""
    args = parse_args()
    project_root = Path.cwd()

    if not args.config.is_file():
        raise FileNotFoundError(args.config)

    with args.config.open("r", encoding="utf-8") as file:
        config: Any = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The experiment configuration must be a YAML mapping.")

    result = run_classical_experiments(
        config,
        project_root=project_root,
    )

    print("\nClassical baseline experiments completed.")
    print(f"Experiments: {len(result['runs'])}")
    print(f"Metrics: {result['metrics_csv']}")
    print(f"Detailed results: {result['results_json']}")


if __name__ == "__main__":
    main()
