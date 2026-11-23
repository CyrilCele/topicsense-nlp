"""Download and persist the 20 Newsgroups benchmark dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.datasets import fetch_20newsgroups


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


def download_20newsgroups(
    output_dir: Path,
    cache_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    """
    Download the official train and test partitions.

    Original document text is preserved without removing headers,
    footers, or quoted material.

    Args:
        output_dir: Directory in which JSONL files are stored.
        cache_dir: Directory used by scikit-learn for downloaded data.
        force: Whether to overwrite existing output files.

    Returns:
        A summary containing the number of documents in each split.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, int] = {}
    target_names: list[str] | None = None

    for split in ("train", "test"):
        output_path = output_dir / f"{split}.jsonl"

        if output_path.exists() and not force:
            raise FileExistsError(
                f"{output_path} already exists. "
                "Use --force to overwrite existing files."
            )

        dataset = fetch_20newsgroups(
            subset=split,
            data_home=str(cache_dir),
            shuffle=False,
            remove=(),
            download_if_missing=True,
        )

        target_names = list(dataset.target_names)

        with output_path.open("w", encoding="utf-8") as file:
            for index, (document, target) in enumerate(
                zip(dataset.data, dataset.target, strict=True)
            ):
                record = {
                    "document_id": f"{split}_{index:06d}",
                    "text": document,
                    "label": int(target),
                    "label_name": target_names[int(target)],
                    "split": split,
                }

                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        split_counts[split] = len(dataset.data)

    metadata = {
        "dataset": "20 Newsgroups",
        "source": "scikit-learn fetch_20newsgroups",
        "source_url": (
            "https://scikit-learn.org/stable/"
            "datasets/real_world.html#the-20-newsgroups-text-dataset"
        ),
        "task_type": "multiclass_text_classification",
        "number_of_classes": len(target_names or []),
        "classes": target_names,
        "split_counts": split_counts,
        "total_documents": sum(split_counts.values()),
        "text_modified_during_acquisition": False,
        "note": (
            "Original text is preserved. Check dataset rights and "
            "licensing before redistribution."
        ),
    }

    metadata_path = output_dir / "metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    return metadata


def main() -> None:
    """Run the dataset acquisition command."""
    parser = argparse.ArgumentParser(
        description="Download the 20 Newsgroups dataset."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset files.",
    )

    args = parser.parse_args()

    output_dir = PROJECT_ROOT / "data" / "raw" / "20newsgroups"
    cache_dir = (
        PROJECT_ROOT / "data" / "external" / "20newsgroups_cache"
    )

    metadata = download_20newsgroups(
        output_dir=output_dir,
        cache_dir=cache_dir,
        force=args.force,
    )

    print("20 Newsgroups acquisition completed.")
    print(f"Training documents: {metadata['split_counts']['train']}")
    print(f"Test documents: {metadata['split_counts']['test']}")
    print(f"Total documents: {metadata['total_documents']}")
    print(f"Classes: {metadata['number_of_classes']}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
