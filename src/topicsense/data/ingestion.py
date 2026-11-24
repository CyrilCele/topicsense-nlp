"""Dataset acquisition and raw dataset preservation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sklearn
from sklearn.datasets import fetch_20newsgroups


def calculate_sha256(text: str) -> str:
    """Calculate the SHA-256 hash of a text document."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def calculate_file_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_20newsgroups(
    *,
    data_home: Path,
    output_path: Path,
    manifest_path: Path,
    subsets: tuple[str, ...] = ("train", "test"),
    shuffle: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Download and preserve the original 20 Newsgroups dataset.

    The original train/test partitions are retained. No text cleaning,
    header removal, tokenisation or other preprocessing is performed.
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Raw dataset already exists: {output_path}. "
            "Use --overwrite to replace it deliberately."
        )

    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Dataset manifest already exists: {manifest_path}. "
            "Use --overwrite to replace it deliberately."
        )

    if set(subsets) != {"train", "test"} or len(subsets) != 2:
        raise ValueError(
            "The primary benchmark must preserve both original "
            "train and test partitions."
        )

    data_home.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    class_distribution: dict[str, Counter[str]] = {}
    source_split_counts: Counter[str] = Counter()
    unique_content_hashes: set[str] = set()

    total_documents = 0
    target_names: list[str] | None = None

    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with temporary_output.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            for subset in subsets:
                dataset = fetch_20newsgroups(
                    subset=subset,
                    data_home=str(data_home),
                    remove=(),
                    shuffle=shuffle,
                    download_if_missing=True,
                )

                current_target_names = list(dataset.target_names)

                if target_names is None:
                    target_names = current_target_names
                elif target_names != current_target_names:
                    raise ValueError(
                        "The target category mappings differ between "
                        "the original training and test partitions."
                    )

                class_distribution[subset] = Counter()

                for index, (text, target) in enumerate(
                    zip(dataset.data, dataset.target, strict=True)
                ):
                    label_id = int(target)
                    label_name = current_target_names[label_id]

                    document_hash = calculate_sha256(text)

                    record = {
                        "record_id": f"20newsgroups-{subset}-{index:05d}",
                        "dataset": "20newsgroups",
                        "source_split": subset,
                        "text": text,
                        "label": label_name,
                        "label_id": label_id,
                        "content_sha256": document_hash,
                    }

                    output_file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    class_distribution[subset][label_name] += 1
                    source_split_counts[subset] += 1
                    unique_content_hashes.add(document_hash)
                    total_documents += 1

        temporary_output.replace(output_path)

    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    if target_names is None:
        raise RuntimeError("The dataset loader returned no categories.")

    manifest: dict[str, Any] = {
        "dataset": "20newsgroups",
        "display_name": "20 Newsgroups",
        "source_url": ("https://people.csail.mit.edu/jrennie/20Newsgroups/"),
        "retrieval_method": ("sklearn.datasets.fetch_20newsgroups"),
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "total_documents": total_documents,
        "number_of_classes": len(target_names),
        "target_names": target_names,
        "source_split_counts": dict(source_split_counts),
        "class_distribution": {
            split: dict(counts) for split, counts in class_distribution.items()
        },
        "unique_content_hashes": len(unique_content_hashes),
        "exact_duplicate_count": (total_documents - len(unique_content_hashes)),
        "preprocessing_applied": [],
        "loader_remove_options": [],
        "original_split_preserved": True,
        "raw_data_file": output_path.name,
        "raw_data_sha256": calculate_file_sha256(output_path),
        "licence_status": (
            "Redistribution license requires verification against "
            "the original dataset source."
        ),
    }

    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

    try:
        with temporary_manifest.open(
            "w",
            encoding="utf-8",
        ) as manifest_file:
            json.dump(
                manifest,
                manifest_file,
                indent=4,
                ensure_ascii=False,
            )
            manifest_file.write("\n")

        temporary_manifest.replace(manifest_path)

    except Exception:
        temporary_manifest.unlink(missing_ok=True)
        raise

    return manifest
