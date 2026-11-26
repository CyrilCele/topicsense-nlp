"""Command-line entry point for transformer experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from topicsense.training.transformers import run_transformer_experiments


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Transformer configuration must be a YAML mapping.")

    return config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TopicSense-NLP transformer experiments."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/transformers.yaml"),
        help="Path to the transformer YAML configuration.",
    )

    parser.add_argument(
        "--experiment",
        choices=["all", "distilbert", "frozen_embeddings"],
        default="all",
        help="Select which transformer experiment to run.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    project_root = Path.cwd().resolve()
    config = load_config(project_root / arguments.config)

    results = run_transformer_experiments(
        config,
        project_root=project_root,
        experiment=arguments.experiment,
    )

    for experiment_name, result in results.items():
        print(f"\n{experiment_name}:")
        print(result["summary"])


if __name__ == "__main__":
    main()
