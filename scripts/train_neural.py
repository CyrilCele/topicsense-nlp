"""Command-line entry point for neural-network experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from topicsense.training.neural import run_neural_experiments


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate TopicSense-NLP neural models."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/neural_networks.yaml"),
        help="Path to the neural-network YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Load configuration and execute neural experiments."""
    args = parse_args()
    project_root = Path.cwd()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        config: Any = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The experiment configuration must be a YAML mapping.")

    result = run_neural_experiments(
        config,
        project_root=project_root,
    )

    print("\nNeural-network experiments completed.")
    print(f"Experiments: {len(result['runs'])}")
    print(f"Metrics: {result['metrics_csv']}")
    print(f"Detailed results: {result['results_json']}")


if __name__ == "__main__":
    main()
