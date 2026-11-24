"""Build reproducible manifests for train-test content contamination audits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MIN_SUBSTANTIVE_WORDS = 5
MIN_SUBSTANTIVE_CHARACTERS = 20


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_body(text: str) -> str:
    """Normalize message-body text using the body-audit rules."""
    import unicodedata

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def extract_body(text: str) -> str:
    """Extract the message body, falling back to the complete text."""
    _, separator, body = text.partition("\n\n")
    return body if separator else text


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Load JSONL records indexed by their unique record IDs."""
    records: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            record_id = record.get("record_id")

            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"Missing or invalid record_id on line {line_number}.")

            if record_id in records:
                raise ValueError(f"Duplicate record_id found: {record_id}")

            records[record_id] = record

    return records


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)

    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}.")

    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically, creating parent directories when necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def validate_record(
    record_id: str,
    record: dict[str, Any],
) -> tuple[str, str, str]:
    """Validate and return a record's split, label, and normalized body."""
    source_split = record.get("source_split")
    label = record.get("label")
    text = record.get("text")

    if source_split not in {"train", "test"}:
        raise ValueError(f"Unexpected source_split for {record_id}: {source_split!r}")

    if not isinstance(label, str) or not label:
        raise ValueError(f"Missing or invalid label for {record_id}.")

    if not isinstance(text, str):
        raise ValueError(f"Missing or invalid text for {record_id}.")

    normalized_body = normalize_body(extract_body(text))
    return source_split, label, normalized_body


def build_contamination_manifests(
    *,
    raw_path: Path,
    body_audit_path: Path,
    leakage_report_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build exact-duplicate sensitivity manifests and audit metadata."""
    records = load_jsonl(raw_path)
    body_audit = load_json(body_audit_path)
    leakage_report = load_json(leakage_report_path)

    duplicate_groups = body_audit.get("duplicate_groups")
    if not isinstance(duplicate_groups, list):
        raise ValueError("The body audit must contain a 'duplicate_groups' list.")

    train_exclusion_ids: set[str] = set()
    test_exclusion_ids: set[str] = set()
    substantive_groups: list[dict[str, Any]] = []
    trivial_groups: list[dict[str, Any]] = []

    for group_index, group in enumerate(duplicate_groups):
        members = group.get("members")

        if not isinstance(members, list):
            raise ValueError(
                f"Duplicate group {group_index} has no valid members list."
            )

        validated_members: list[dict[str, str]] = []

        for member in members:
            record_id = member.get("record_id")

            if not isinstance(record_id, str) or record_id not in records:
                raise ValueError(f"Unknown record ID in duplicate group: {record_id!r}")

            source_split, label, normalized_body = validate_record(
                record_id,
                records[record_id],
            )

            validated_members.append(
                {
                    "record_id": record_id,
                    "source_split": source_split,
                    "label": label,
                    "normalized_body": normalized_body,
                }
            )

        partitions = {member["source_split"] for member in validated_members}

        if partitions != {"train", "test"}:
            continue

        normalized_bodies = {member["normalized_body"] for member in validated_members}

        if len(normalized_bodies) != 1:
            raise ValueError(
                "A reported exact-duplicate group contains different "
                f"normalized bodies: {group.get('body_hash')}"
            )

        normalized_body = next(iter(normalized_bodies))
        word_count = len(normalized_body.split())
        character_count = len(normalized_body)

        group_record = {
            "body_hash": group.get("body_hash"),
            "members": [
                {
                    "record_id": member["record_id"],
                    "source_split": member["source_split"],
                    "label": member["label"],
                }
                for member in validated_members
            ],
            "labels": sorted({member["label"] for member in validated_members}),
            "label_agreement": (
                len({member["label"] for member in validated_members}) == 1
            ),
            "normalized_word_count": word_count,
            "normalized_character_count": character_count,
        }

        is_substantive = (
            word_count >= MIN_SUBSTANTIVE_WORDS
            and character_count >= MIN_SUBSTANTIVE_CHARACTERS
        )

        if not is_substantive:
            trivial_groups.append(group_record)
            continue

        substantive_groups.append(group_record)

        for member in validated_members:
            if member["source_split"] == "train":
                train_exclusion_ids.add(member["record_id"])
            else:
                test_exclusion_ids.add(member["record_id"])

    if not substantive_groups:
        raise ValueError(
            "No substantive cross-partition exact-duplicate groups found. "
            "Review the body-audit report before generating manifests."
        )

    source_split_counts = Counter(record["source_split"] for record in records.values())

    near_duplicate_pairs = leakage_report.get("duplicate_analysis", {}).get(
        "near_duplicate_pairs", []
    )

    cross_partition_near_duplicate_count = sum(
        1 for pair in near_duplicate_pairs if pair.get("cross_partition") is True
    )

    manifest = {
        "manifest_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "method": "substantive_normalized_exact_body_duplicates",
        "normalization": [
            "Unicode NFKC",
            "case folding",
            "whitespace normalization",
        ],
        "substantive_body_threshold": {
            "minimum_words": MIN_SUBSTANTIVE_WORDS,
            "minimum_characters": MIN_SUBSTANTIVE_CHARACTERS,
        },
        "source_files": {
            "raw_dataset": str(raw_path),
            "raw_dataset_sha256": sha256_file(raw_path),
            "body_audit": str(body_audit_path),
            "body_audit_sha256": sha256_file(body_audit_path),
            "leakage_report": str(leakage_report_path),
            "leakage_report_sha256": sha256_file(leakage_report_path),
        },
        "dataset": {
            "record_count": len(records),
            "official_train_count": source_split_counts["train"],
            "official_test_count": source_split_counts["test"],
        },
        "summary": {
            "substantive_cross_partition_exact_duplicate_groups": len(
                substantive_groups
            ),
            "substantive_training_documents_implicated": len(train_exclusion_ids),
            "substantive_test_documents_implicated": len(test_exclusion_ids),
            "trivial_cross_partition_exact_duplicate_groups": len(trivial_groups),
            "cross_partition_near_duplicate_pairs_not_excluded": (
                cross_partition_near_duplicate_count
            ),
            "clean_test_document_count": (
                source_split_counts["test"] - len(test_exclusion_ids)
            ),
            "decontaminated_training_document_count": (
                source_split_counts["train"] - len(train_exclusion_ids)
            ),
        },
        "substantive_exact_duplicate_groups": substantive_groups,
        "trivial_exact_duplicate_groups": trivial_groups,
        "near_duplicate_policy": (
            "Near-duplicate pairs are reported separately and are not "
            "excluded by these exact-duplicate sensitivity manifests."
        ),
        "evaluation_tracks": {
            "official_benchmark": {
                "training": "Original official training partition.",
                "evaluation": "Original official test partition.",
                "exclusions": [],
            },
            "clean_test_sensitivity": {
                "training": "Original official training partition.",
                "evaluation": (
                    "Official test partition excluding test documents "
                    "with substantive exact-body duplicates in training."
                ),
                "excluded_test_record_ids": sorted(test_exclusion_ids),
            },
            "decontaminated_training_sensitivity": {
                "training": (
                    "Official training partition excluding training "
                    "documents with substantive exact-body duplicates "
                    "in the official test partition."
                ),
                "evaluation": "Original official test partition.",
                "excluded_training_record_ids": sorted(train_exclusion_ids),
            },
        },
    }

    write_json(output_dir / "contamination_manifest.json", manifest)

    write_json(
        output_dir / "clean_test_ids.json",
        {
            "description": (
                "Test record IDs excluded from the clean-test sensitivity "
                "evaluation because their normalized bodies occur in "
                "the official training partition."
            ),
            "record_ids": sorted(test_exclusion_ids),
        },
    )

    write_json(
        output_dir / "decontaminated_train_ids.json",
        {
            "description": (
                "Training record IDs excluded from the decontaminated-"
                "training sensitivity experiment because their normalized "
                "bodies occur in the official test partition."
            ),
            "record_ids": sorted(train_exclusion_ids),
        },
    )

    return manifest
