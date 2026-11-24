"""Build train-test contamination sensitivity manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from topicsense.data.contamination import build_contamination_manifests


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build reproducible sensitivity manifests from the "
            "20 Newsgroups duplicate-body audit."
        )
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path("data/raw/20newsgroups/20newsgroups_raw.jsonl"),
    )
    parser.add_argument(
        "--body-audit-path",
        type=Path,
        default=Path("results/logs/dataset_body_duplicate_audit.json"),
    )
    parser.add_argument(
        "--leakage-report-path",
        type=Path,
        default=Path("results/logs/dataset_leakage_analysis.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/20newsgroups"),
    )
    return parser.parse_args()


def main() -> None:
    """Build manifests and print a concise summary."""
    args = parse_args()

    manifest = build_contamination_manifests(
        raw_path=args.raw_path,
        body_audit_path=args.body_audit_path,
        leakage_report_path=args.leakage_report_path,
        output_dir=args.output_dir,
    )

    print("Contamination manifests generated successfully.")
    print(f"Output directory: {args.output_dir}")
    print("Summary:")

    for key, value in manifest["summary"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
