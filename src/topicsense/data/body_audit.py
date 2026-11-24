"""Second-stage audit for exact normalized document-body duplicates."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


def normalize_body(text: str) -> str:
    """Normalize text for exact body-duplicate comparison."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.casefold()
    return " ".join(normalized.split())


def extract_body(text: str) -> str:
    """Extract the message body after the first header/body separator."""
    _, separator, body = text.partition("\n\n")

    if not separator:
        return text

    return body


def calculate_body_hash(body: str) -> str:
    """Calculate SHA-256 over the normalized message body."""
    normalized = normalize_body(body)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from disk."""
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {path}"
                )

            records.append(record)

    return records


def audit_body_duplicates(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Identify exact normalized body duplicates across source partitions."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        record_id = record.get("record_id")
        source_split = record.get("source_split")
        label = record.get("label")
        raw_text = record.get("text")

        if not isinstance(record_id, str) or not record_id:
            raise ValueError("Every record must have a non-empty record_id.")

        if not isinstance(source_split, str) or not source_split:
            raise ValueError(f"Record {record_id!r} has no valid source_split value.")

        if not isinstance(raw_text, str):
            raise ValueError(f"Record {record_id!r} has no valid text field.")

        if not isinstance(label, str) or not label:
            raise ValueError(f"Record {record_id!r} has no valid label.")

        body = extract_body(raw_text)
        normalized_body = normalize_body(body)

        if not normalized_body:
            continue

        body_hash = calculate_body_hash(body)

        groups[body_hash].append(
            {
                "record_id": record_id,
                "label": label,
                "source_split": source_split,
                "body_hash": body_hash,
            }
        )

    duplicate_groups: list[dict[str, Any]] = []

    for body_hash, members in sorted(groups.items()):
        if len(members) < 2:
            continue

        partitions = sorted({member["source_split"] for member in members})
        labels = sorted({member["label"] for member in members})

        duplicate_groups.append(
            {
                "body_hash": body_hash,
                "document_count": len(members),
                "partitions": partitions,
                "labels": labels,
                "cross_partition": len(partitions) > 1,
                "label_agreement": len(labels) == 1,
                "members": sorted(
                    members,
                    key=lambda member: (
                        member["source_split"],
                        member["record_id"],
                    ),
                ),
            }
        )

    cross_partition_groups = [
        group for group in duplicate_groups if group["cross_partition"]
    ]

    conflicting_label_groups = [
        group for group in duplicate_groups if not group["label_agreement"]
    ]

    cross_partition_conflicting_groups = [
        group for group in cross_partition_groups if not group["label_agreement"]
    ]

    analysis = {
        "method": "normalized_exact_document_body_sha256",
        "normalization": [
            "Unicode NFKC",
            "case folding",
            "whitespace normalization",
        ],
        "original_records_modified": False,
        "records_analyzed": len(records),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_document_count": sum(
            group["document_count"] for group in duplicate_groups
        ),
        "cross_partition_duplicate_group_count": len(cross_partition_groups),
        "cross_partition_duplicate_document_count": sum(
            group["document_count"] for group in cross_partition_groups
        ),
        "conflicting_label_group_count": len(conflicting_label_groups),
        "cross_partition_conflicting_label_group_count": len(
            cross_partition_conflicting_groups
        ),
    }

    return {
        "analysis": analysis,
        "duplicate_groups": duplicate_groups,
    }


def run_body_audit(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run the body-duplicate audit and write its JSON report."""
    input_file = Path(input_path)
    output_file = Path(output_path)

    records = load_jsonl(input_file)
    report = audit_body_duplicates(records)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return report
