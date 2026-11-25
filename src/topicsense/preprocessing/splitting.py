"""Leakage-aware train/validation splitting for the 20 Newsgroups benchmark."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
                raise ValueError(f"Expected a JSON object on line {line_number}.")

            records.append(record)

    return records


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")

    return payload


class _UnionFind:
    """Build connected components from duplicate relationships."""

    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = dict.fromkeys(values, 0)

    def find(self, value: str) -> str:
        parent = self.parent[value]

        if parent != value:
            self.parent[value] = self.find(parent)

        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)

        if root_left == root_right:
            return

        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left

        self.parent[root_right] = root_left

        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _build_training_groups(
    training_ids: list[str],
    body_audit: dict[str, Any],
    leakage_report: dict[str, Any],
) -> tuple[dict[str, str], dict[str, int]]:
    """Connect training records linked by exact-body or reported near duplicates."""
    training_id_set = set(training_ids)
    union_find = _UnionFind(training_ids)

    exact_edges = 0
    near_edges = 0

    duplicate_groups = body_audit.get("duplicate_groups")

    if not isinstance(duplicate_groups, list):
        raise ValueError("Body audit must contain a duplicate_groups list.")

    for group in duplicate_groups:
        members = group.get("members", [])

        if not isinstance(members, list):
            raise ValueError("Each body-audit duplicate group must contain members.")

        group_ids: list[str] = []

        for member in members:
            if not isinstance(member, dict):
                continue

            record_id = member.get("record_id")

            if isinstance(record_id, str) and record_id in training_id_set:
                group_ids.append(record_id)

        if len(group_ids) > 1:
            for record_id in group_ids[1:]:
                union_find.union(group_ids[0], record_id)
                exact_edges += 1

    duplicate_analysis = leakage_report.get("duplicate_analysis", {})
    near_pairs = duplicate_analysis.get("near_duplicate_pairs", [])

    if not isinstance(near_pairs, list):
        raise ValueError("Leakage report near_duplicate_pairs must be a list.")

    for pair in near_pairs:
        if not isinstance(pair, dict):
            continue

        left = pair.get("left_record_id")
        right = pair.get("right_record_id")

        if (
            isinstance(left, str)
            and isinstance(right, str)
            and left in training_id_set
            and right in training_id_set
        ):
            union_find.union(left, right)
            near_edges += 1

    components: dict[str, list[str]] = defaultdict(list)

    for record_id in training_ids:
        components[union_find.find(record_id)].append(record_id)

    group_by_id: dict[str, str] = {}

    for members in components.values():
        stable_digest = hashlib.sha256(
            "\n".join(sorted(members)).encode("utf-8")
        ).hexdigest()[:16]

        group_id = f"dup-{stable_digest}"

        for record_id in members:
            group_by_id[record_id] = group_id

    diagnostics = {
        "exact_duplicate_links_considered": exact_edges,
        "near_duplicate_links_considered": near_edges,
        "connected_component_count": len(components),
        "multi_document_component_count": sum(
            len(members) > 1 for members in components.values()
        ),
        "largest_component_size": max(map(len, components.values()), default=0),
    }

    return group_by_id, diagnostics


def _distribution(labels: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(labels).items()))


def _make_split(
    *,
    records_by_id: dict[str, dict[str, Any]],
    selected_ids: list[str],
    group_by_id: dict[str, str],
    validation_size: float,
    random_seed: int,
    track_name: str,
) -> dict[str, Any]:
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")

    inverse_fraction = 1.0 / validation_size
    n_splits = round(inverse_fraction)

    if n_splits < 2 or not np.isclose(inverse_fraction, n_splits):
        raise ValueError(
            "StratifiedGroupKFold requires validation_size to be the "
            "reciprocal of an integer (for example, 0.20)."
        )

    ordered_ids = sorted(selected_ids)
    labels = [records_by_id[record_id]["label"] for record_id in ordered_ids]
    groups = [group_by_id[record_id] for record_id in ordered_ids]

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )

    dummy_features = np.zeros((len(ordered_ids), 1), dtype=np.uint8)
    folds = list(splitter.split(dummy_features, labels, groups))

    train_indices, validation_indices = folds[0]

    train_ids = [ordered_ids[index] for index in train_indices]
    validation_ids = [ordered_ids[index] for index in validation_indices]

    train_groups = {group_by_id[record_id] for record_id in train_ids}
    validation_groups = {group_by_id[record_id] for record_id in validation_ids}

    overlap = train_groups & validation_groups

    if overlap:
        raise RuntimeError("A duplicate-connected group crossed the split boundary.")

    train_labels = [records_by_id[record_id]["label"] for record_id in train_ids]
    validation_labels = [
        records_by_id[record_id]["label"] for record_id in validation_ids
    ]

    expected_classes = set(labels)

    if (
        set(train_labels) != expected_classes
        or set(validation_labels) != expected_classes
    ):
        raise ValueError(
            "The group-aware split did not retain every class in both "
            "training and validation. Inspect group structure before proceeding."
        )

    return {
        "track": track_name,
        "method": "StratifiedGroupKFold over duplicate-connected components",
        "validation_size_target": validation_size,
        "fold_count": n_splits,
        "selected_fold": 0,
        "random_seed": random_seed,
        "document_count": len(ordered_ids),
        "training_count": len(train_ids),
        "validation_count": len(validation_ids),
        "training_fraction_actual": len(train_ids) / len(ordered_ids),
        "validation_fraction_actual": len(validation_ids) / len(ordered_ids),
        "training_class_distribution": _distribution(train_labels),
        "validation_class_distribution": _distribution(validation_labels),
        "training_group_count": len(train_groups),
        "validation_group_count": len(validation_groups),
        "group_overlap_count": len(overlap),
        "training_record_ids": sorted(train_ids),
        "validation_record_ids": sorted(validation_ids),
    }


def create_preprocessing_splits(
    *,
    raw_dataset_path: Path,
    body_audit_path: Path,
    leakage_report_path: Path,
    decontaminated_train_ids_path: Path,
    output_dir: Path,
    validation_size: float = 0.20,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Create official and decontaminated train/validation split manifests.

    The official test partition is never passed to the splitter. Duplicate
    relationships are used only to keep connected training records together.
    """
    records = _read_jsonl(raw_dataset_path)
    records_by_id: dict[str, dict[str, Any]] = {}

    for record in records:
        record_id = record.get("record_id")

        if not isinstance(record_id, str) or not record_id:
            raise ValueError("Every raw record must have a non-empty record_id.")

        if record_id in records_by_id:
            raise ValueError(f"Duplicate record_id found: {record_id}")

        if record.get("source_split") not in {"train", "test"}:
            raise ValueError(f"Invalid source_split for {record_id}.")

        if not isinstance(record.get("label"), str):
            raise ValueError(f"Invalid label for {record_id}.")

        records_by_id[record_id] = record

    body_audit = _read_json(body_audit_path)
    leakage_report = _read_json(leakage_report_path)
    decontaminated_payload = _read_json(decontaminated_train_ids_path)

    excluded_ids = decontaminated_payload.get("record_ids")

    if not isinstance(excluded_ids, list) or not all(
        isinstance(record_id, str) for record_id in excluded_ids
    ):
        raise ValueError("Decontaminated training IDs must contain a record_ids list.")

    official_train_ids = [
        record_id
        for record_id, record in records_by_id.items()
        if record["source_split"] == "train"
    ]

    official_test_ids = [
        record_id
        for record_id, record in records_by_id.items()
        if record["source_split"] == "test"
    ]

    unknown_exclusions = set(excluded_ids) - set(records_by_id)

    non_training_exclusions = {
        record_id
        for record_id in excluded_ids
        if record_id in records_by_id
        and records_by_id[record_id]["source_split"] != "train"
    }

    if unknown_exclusions or non_training_exclusions:
        raise ValueError(
            "Decontamination exclusions contain unknown or non-training IDs."
        )

    group_by_id, group_diagnostics = _build_training_groups(
        official_train_ids,
        body_audit,
        leakage_report,
    )

    official_split = _make_split(
        records_by_id=records_by_id,
        selected_ids=official_train_ids,
        group_by_id=group_by_id,
        validation_size=validation_size,
        random_seed=random_seed,
        track_name="official_training",
    )

    decontaminated_ids = sorted(set(official_train_ids) - set(excluded_ids))

    decontaminated_split = _make_split(
        records_by_id=records_by_id,
        selected_ids=decontaminated_ids,
        group_by_id=group_by_id,
        validation_size=validation_size,
        random_seed=random_seed,
        track_name="decontaminated_training",
    )

    split_payloads: dict[str, dict[str, Any]] = {
        "official_training": official_split,
        "decontaminated_training": decontaminated_split,
    }

    result = {
        "dataset": "20newsgroups",
        "random_seed": random_seed,
        "validation_size_target": validation_size,
        "official_test_count": len(official_test_ids),
        "grouping": {
            "basis": [
                "normalized exact message-body duplicate groups",
                (
                    "MinHash candidate pairs verified by exact Jaccard "
                    "at the audit threshold"
                ),
            ],
            **group_diagnostics,
        },
        "evaluation_tracks": {
            "official_benchmark": {
                "development_split": "official_training",
                "evaluation_partition": "all official test records",
            },
            "clean_test_sensitivity": {
                "development_split": "official_training",
                "evaluation_partition": (
                    "official test excluding records listed in clean_test_ids.json"
                ),
                "clean_test_ids_path": "data/interim/20newsgroups/clean_test_ids.json",
            },
            "decontaminated_training_sensitivity": {
                "development_split": "decontaminated_training",
                "evaluation_partition": "all official test records",
                "decontaminated_train_ids_path": str(decontaminated_train_ids_path),
            },
        },
        "splits": split_payloads,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for key, payload in split_payloads.items():
        output_path = output_dir / f"{key}_split.json"
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    (output_dir / "preprocessing_split_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return result
