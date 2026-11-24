"""Dataset integrity and quality validation utilities."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "record_id",
    "dataset",
    "source_split",
    "text",
    "label",
    "label_id",
    "content_sha256",
}


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


def validate_dataset(
    *,
    dataset_path: Path,
    manifest_path: Path,
    expected_documents: int,
    expected_classes: int,
) -> dict[str, Any]:
    """Validate dataset records against the acquisition manifest."""
    errors: list[str] = []
    warnings: list[str] = []

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    target_names = manifest.get("target_names")

    if not isinstance(target_names, list):
        raise ValueError("Manifest must contain a target_names list.")

    if len(target_names) != expected_classes:
        errors.append(
            f"Expected {expected_classes} categories, "
            f"but manifest contains {len(target_names)}."
        )

    if len(set(target_names)) != len(target_names):
        errors.append("Manifest contains duplicate category names.")

    if manifest.get("total_documents") != expected_documents:
        errors.append(
            "Manifest document count does not match the configured "
            f"expectation of {expected_documents}."
        )

    source_split_counts: Counter[str] = Counter()
    class_distribution: dict[str, Counter[str]] = {
        "train": Counter(),
        "test": Counter(),
    }

    record_ids: set[str] = set()
    content_hashes: set[str] = set()

    total_records = 0
    empty_documents = 0
    invalid_records = 0
    document_hash_mismatches = 0
    duplicate_record_ids = 0

    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"Invalid JSON on line {line_number}.")
                invalid_records += 1
                continue

            if not isinstance(record, dict):
                errors.append(f"Record on line {line_number} is not a JSON object.")
                invalid_records += 1
                continue

            total_records += 1

            missing_fields = REQUIRED_FIELDS - record.keys()

            if missing_fields:
                errors.append(
                    f"Record on line {line_number} is missing fields: "
                    f"{sorted(missing_fields)}."
                )
                invalid_records += 1
                continue

            record_id = record["record_id"]
            text = record["text"]
            label = record["label"]
            label_id = record["label_id"]
            source_split = record["source_split"]
            document_hash = record["content_sha256"]

            if not isinstance(record_id, str) or not record_id:
                errors.append(f"Invalid record ID on line {line_number}.")
                invalid_records += 1
                continue

            if record_id in record_ids:
                duplicate_record_ids += 1
            else:
                record_ids.add(record_id)

            if not isinstance(text, str):
                errors.append(f"Text is not a string on line {line_number}.")
                invalid_records += 1
                continue

            if not text.strip():
                empty_documents += 1

            if not isinstance(label, str) or label not in target_names:
                errors.append(
                    f"Invalid category label on line {line_number}: {label!r}."
                )
                invalid_records += 1
                continue

            if type(label_id) is not int:
                errors.append(f"Invalid label ID on line {line_number}.")
                invalid_records += 1
                continue

            if not 0 <= label_id < len(target_names):
                errors.append(f"Out-of-range label ID on line {line_number}.")
                invalid_records += 1
                continue

            if target_names[label_id] != label:
                errors.append(f"Label mapping mismatch on line {line_number}.")
                invalid_records += 1
                continue

            if source_split not in ("train", "test"):
                errors.append(f"Invalid source partition on line {line_number}.")
                invalid_records += 1
                continue

            if record["dataset"] != manifest.get("dataset"):
                errors.append(f"Dataset name mismatch on line {line_number}.")
                invalid_records += 1
                continue

            actual_hash = calculate_sha256(text)

            if actual_hash != document_hash:
                document_hash_mismatches += 1

            content_hashes.add(actual_hash)
            source_split_counts[source_split] += 1
            class_distribution[source_split][label] += 1

    exact_duplicate_count = total_records - len(content_hashes)

    if duplicate_record_ids:
        errors.append(f"Found {duplicate_record_ids} duplicate record IDs.")

    if empty_documents:
        warnings.append(f"Found {empty_documents} empty or whitespace-only documents.")

    if document_hash_mismatches:
        errors.append(f"Found {document_hash_mismatches} document hash mismatches.")

    if invalid_records:
        warnings.append(
            f"Found {invalid_records} records with structural or "
            "field validation errors."
        )

    if total_records != expected_documents:
        errors.append(
            f"Expected {expected_documents} records, but read {total_records}."
        )

    expected_split_counts = manifest.get("source_split_counts", {})

    if dict(source_split_counts) != expected_split_counts:
        errors.append("Actual source partition counts differ from the manifest.")

    expected_distribution = manifest.get("class_distribution", {})

    actual_distribution = {
        split: dict(counts) for split, counts in class_distribution.items()
    }

    if actual_distribution != expected_distribution:
        errors.append("Actual class distribution differ from the manifest.")

    if len(content_hashes) != manifest.get("unique_content_hashes"):
        errors.append("Unique document hash count differs from the manifest.")

    if exact_duplicate_count != manifest.get("exact_duplicate_count"):
        errors.append("Exact duplicate count differs from the manifest.")

    actual_file_path = calculate_file_sha256(dataset_path)

    if actual_file_path != manifest.get("raw_data_sha256"):
        errors.append("Dataset file SHA-256 does not match the acquisition manifest.")

    if not manifest.get("original_split_preserved", False):
        errors.append("The manifest does not confirm original split preservation.")

    if manifest.get("preprocessing_applied") != []:
        warnings.append("The manifest indicates that preprocessing was applied.")

    return {
        "dataset": manifest.get("dataset"),
        "validation_status": "passed" if not errors else "failed",
        "total_records": total_records,
        "number_of_classes": len(target_names),
        "source_split_counts": dict(source_split_counts),
        "class_distribution": actual_distribution,
        "empty_documents": empty_documents,
        "invalid_records": invalid_records,
        "duplicate_record_ids": duplicate_record_ids,
        "unique_content_hashes": len(content_hashes),
        "exact_duplicate_count": exact_duplicate_count,
        "document_hash_mismatches": document_hash_mismatches,
        "raw_data_sha256": actual_file_path,
        "errors": errors,
        "warnings": warnings,
    }
