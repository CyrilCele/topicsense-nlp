"""Duplicate detection and metadata leakage analysis."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from email.parser import Parser
from pathlib import Path
from typing import Any

_MINHASH_PRIME = (1 << 61) - 1
_MESSAGE_ID_PATTERN = re.compile(r"<[^<>]+>")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def hash_identifier(value: str) -> str:
    """Create a non-reversible SHA-256 identifier."""
    normalized = value.strip().lower()

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_headers(text: str) -> dict[str, Any]:
    """Extract selected email headers without retaining raw identities."""
    parts = re.split(r"\r?\n\r?\n", text, maxsplit=1)
    header_text = parts[0]

    message = Parser().parsestr(header_text, headersonly=True)

    sender = message.get("From", "")
    message_id = message.get("Message-ID", "")
    references = message.get("References", "")
    in_reply_to = message.get("In-Reply-To", "")
    newsgroups = message.get("Newsgroups", "")

    sender_hash = hash_identifier(sender) if sender.strip() else None

    message_id_match = _MESSAGE_ID_PATTERN.search(message_id)
    message_id_hash = (
        hash_identifier(message_id_match.group()) if message_id_match else None
    )

    reference_values = _MESSAGE_ID_PATTERN.findall(
        references
    ) + _MESSAGE_ID_PATTERN.findall(in_reply_to)

    reference_hashes = {hash_identifier(value) for value in reference_values}

    newsgroup_values = {
        value.strip().lower() for value in newsgroups.split(",") if value.strip()
    }

    return {
        "sender_hash": sender_hash,
        "message_id_hash": message_id_hash,
        "reference_hashes": reference_hashes,
        "newsgroup_values": newsgroup_values,
        "has_sender": bool(sender_hash),
        "has_message_id": bool(message_id_hash),
        "has_references": bool(reference_hashes),
        "has_newsgroups": bool(newsgroup_values),
    }


def extract_body(text: str) -> str:
    """Extract the message body while preserving its original content."""
    parts = re.split(r"\r?\n\r?\n", text, maxsplit=1)

    if len(parts) == 2:
        return parts[1]

    return text


def generate_shingles(
    text: str,
    *,
    shingle_size: int,
) -> set[int]:
    """Generate deterministic hashes of consecutive word shingles."""
    tokens = _TOKEN_PATTERN.findall(text.lower())

    if len(tokens) < shingle_size:
        return set()

    shingles = set()

    for index in range(len(tokens) - shingle_size + 1):
        shingle = " ".join(tokens[index : index + shingle_size])
        digest = hashlib.blake2b(
            shingle.encode("utf-8"),
            digest_size=8,
        ).digest()
        shingles.add(int.from_bytes(digest, byteorder="big"))

    return shingles


def generate_minhash_signature(
    shingles: set[int],
    coefficients: list[tuple[int, int]],
) -> tuple[int, ...]:
    """Generate a deterministic MinHash signature."""
    signature = [_MINHASH_PRIME] * len(coefficients)

    for shingle_hash in shingles:
        for index, (coefficient_a, coefficient_b) in enumerate(coefficients):
            value = (coefficient_a * shingle_hash + coefficient_b) % _MINHASH_PRIME

            if value < signature[index]:
                signature[index] = value

    return tuple(signature)


def discover_candidate_pairs(
    documents: list[dict[str, Any]],
    *,
    shingle_size: int,
    num_permutations: int,
    num_bands: int,
    max_bucket_size: int,
    random_seed: int,
) -> tuple[set[tuple[int, int]], int]:
    """Discover potential near-duplicate pairs using MinHash LSH."""
    if num_permutations % num_bands != 0:
        raise ValueError(
            "The number of permutations must be divisible by the number of LSH bands."
        )

    random_generator = random.Random(random_seed)

    coefficients = [
        (
            random_generator.randrange(1, _MINHASH_PRIME),
            random_generator.randrange(0, _MINHASH_PRIME),
        )
        for _ in range(num_permutations)
    ]

    rows_per_band = num_permutations // num_bands

    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)

    for document_index, document in enumerate(documents):
        body = extract_body(document["text"])

        shingles = generate_shingles(
            body,
            shingle_size=shingle_size,
        )

        if not shingles:
            continue

        signature = generate_minhash_signature(
            shingles,
            coefficients,
        )

        for band_index in range(num_bands):
            start = band_index * rows_per_band
            end = start + rows_per_band
            bucket_key = (band_index, signature[start:end])

            buckets[bucket_key].append(document_index)

    candidate_pairs: set[tuple[int, int]] = set()
    skipped_buckets = 0

    for bucket_documents in buckets.values():
        if len(bucket_documents) > max_bucket_size:
            skipped_buckets += 1
            continue

        for left_position in range(len(bucket_documents)):
            for right_position in range(
                left_position + 1,
                len(bucket_documents),
            ):
                left_index = bucket_documents[left_position]
                right_index = bucket_documents[right_position]

                candidate_pairs.add((left_index, right_index))

    return candidate_pairs, skipped_buckets


def calculate_jaccard_similarity(
    left: set[int],
    right: set[int],
) -> float:
    """Calculate Jaccard similarity between two shingle sets."""
    union_size = len(left | right)

    if union_size == 0:
        return 0.0

    return len(left & right) / union_size


def analyze_header_leakage(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure cross-partition overlap in selected email metadata."""
    split_senders: dict[str, set[str]] = {
        "train": set(),
        "test": set(),
    }

    split_message_ids: dict[str, set[str]] = {
        "train": set(),
        "test": set(),
    }

    split_references: dict[str, set[str]] = {
        "train": set(),
        "test": set(),
    }

    header_counts: dict[str, int] = defaultdict(int)

    newsgroups_matches = 0
    newsgroups_mismatches = 0

    for document in documents:
        metadata = extract_headers(document["text"])
        split = document["source_split"]
        label = document["label"].lower()

        for field in (
            "has_sender",
            "has_message_id",
            "has_references",
            "has_newsgroups",
        ):
            if metadata[field]:
                header_counts[field] += 1

        if metadata["sender_hash"]:
            split_senders[split].add(metadata["sender_hash"])

        if metadata["message_id_hash"]:
            split_message_ids[split].add(metadata["message_id_hash"])

        split_references[split].update(metadata["reference_hashes"])

        if metadata["has_newsgroups"]:
            if label in metadata["newsgroup_values"]:
                newsgroups_matches += 1
            else:
                newsgroups_mismatches += 1

    shared_senders = split_senders["train"] & split_senders["test"]
    shared_message_ids = split_message_ids["train"] & split_message_ids["test"]
    shared_references = split_references["train"] & split_references["test"]

    return {
        "header_presence_counts": dict(header_counts),
        "newsgroups_label_matches": newsgroups_matches,
        "newsgroups_label_mismatches": newsgroups_mismatches,
        "unique_train_senders": len(split_senders["train"]),
        "unique_test_senders": len(split_senders["test"]),
        "shared_senders_across_partitions": len(shared_senders),
        "shared_message_ids_across_partitions": len(shared_message_ids),
        "shared_thread_references_across_partitions": len(shared_references),
        "privacy_note": (
            "Raw sender identities and message identifiers are not "
            "included in this report."
        ),
    }


def analyze_duplicates(
    documents: list[dict[str, Any]],
    *,
    shingle_size: int,
    num_permutations: int,
    num_bands: int,
    similarity_threshold: float,
    random_seed: int,
    max_bucket_size: int = 250,
) -> dict[str, Any]:
    """Detect likely near-duplicate documents using MinHash and Jaccard."""
    candidate_pairs, skipped_buckets = discover_candidate_pairs(
        documents,
        shingle_size=shingle_size,
        num_permutations=num_permutations,
        num_bands=num_bands,
        max_bucket_size=max_bucket_size,
        random_seed=random_seed,
    )

    shingle_cache: dict[int, set[int]] = {}
    near_duplicate_pairs: list[dict[str, Any]] = []

    cross_partition_pairs = 0
    cross_partition_label_disagreements = 0
    total_near_duplicate_pairs = 0

    for left_index, right_index in sorted(candidate_pairs):
        if left_index not in shingle_cache:
            shingle_cache[left_index] = generate_shingles(
                extract_body(documents[left_index]["text"]),
                shingle_size=shingle_size,
            )

        if right_index not in shingle_cache:
            shingle_cache[right_index] = generate_shingles(
                extract_body(documents[right_index]["text"]),
                shingle_size=shingle_size,
            )

        similarity = calculate_jaccard_similarity(
            shingle_cache[left_index],
            shingle_cache[right_index],
        )

        if similarity < similarity_threshold:
            continue

        left_document = documents[left_index]
        right_document = documents[right_index]

        same_partition = left_document["source_split"] == right_document["source_split"]

        if not same_partition:
            cross_partition_pairs += 1

            if left_document["label"] != right_document["label"]:
                cross_partition_label_disagreements += 1

        total_near_duplicate_pairs += 1

        near_duplicate_pairs.append(
            {
                "left_record_id": left_document["record_id"],
                "right_record_id": right_document["record_id"],
                "left_partition": left_document["source_split"],
                "right_partition": right_document["source_split"],
                "left_label": left_document["label"],
                "right_label": right_document["label"],
                "jaccard_similarity": round(similarity, 6),
                "cross_partition": not same_partition,
                "label_agreement": (left_document["label"] == right_document["label"]),
            }
        )

    near_duplicate_pairs.sort(
        key=lambda pair: pair["jaccard_similarity"],
        reverse=True,
    )

    return {
        "method": "MinHash locality-sensitive hashing with exact Jaccard verification",
        "shingle_type": "consecutive word shingles from message bodies",
        "shingle_size": shingle_size,
        "num_permutations": num_permutations,
        "num_bands": num_bands,
        "random_seed": random_seed,
        "similarity_threshold": similarity_threshold,
        "max_bucket_size": max_bucket_size,
        "candidate_pairs_examined": len(candidate_pairs),
        "oversized_buckets_skipped": skipped_buckets,
        "near_duplicate_pair_count": total_near_duplicate_pairs,
        "cross_partition_near_duplicate_pairs": cross_partition_pairs,
        "cross_partition_label_disagreements": (cross_partition_label_disagreements),
        "reported_pairs_limit": 1000,
        "near_duplicate_pairs": near_duplicate_pairs[:1000],
        "limitation": (
            "LSH is approximate. Candidate pairs may be missed, and "
            "similarity is based on word-shingle Jaccard rather than "
            "semantic equivalence."
        ),
    }


def load_documents(dataset_path: Path) -> list[dict[str, Any]]:
    """Load raw dataset records for analysis."""
    documents = []

    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}.") from error

            documents.append(record)

    return documents


def analyze_dataset_leakage(
    *,
    dataset_path: Path,
    shingle_size: int = 5,
    num_permutations: int = 32,
    num_bands: int = 8,
    similarity_threshold: float = 0.85,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Run duplicate and metadata leakage analyses."""
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("Similarity threshold must be in (0, 1].")

    documents = load_documents(dataset_path)

    header_analysis = analyze_header_leakage(documents)

    duplicate_analysis = analyze_duplicates(
        documents,
        shingle_size=shingle_size,
        num_permutations=num_permutations,
        num_bands=num_bands,
        similarity_threshold=similarity_threshold,
        random_seed=random_seed,
    )

    return {
        "dataset": "20newsgroups",
        "total_documents_analyzed": len(documents),
        "duplicate_analysis": duplicate_analysis,
        "metadata_leakage_analysis": header_analysis,
        "interpretation_warning": (
            "Shared authors or thread references do not automatically "
            "prove train/test leakage. They identify dependencies that "
            "must be considered when interpreting evaluation results."
        ),
    }
