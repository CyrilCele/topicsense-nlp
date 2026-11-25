"""Reproducible TextCNN and BiLSTM training and validation experiments."""

from __future__ import annotations

import csv
import json
import random
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from topicsense.evaluation.metrics import calculate_classification_metrics
from topicsense.models.neural import BiLSTM, TextCNN
from topicsense.preprocessing.text import preprocess_text, tokenize_words

PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
PADDING_INDEX = 0
UNKNOWN_INDEX = 1


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")

    return payload


def _load_training_records(path: Path) -> dict[str, dict[str, str]]:
    """Load official training records only."""
    records: dict[str, dict[str, str]] = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}."
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}.")

            if record.get("source_split") != "train":
                continue

            record_id = record.get("record_id")
            text = record.get("text")
            label = record.get("label")

            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"Invalid record ID on line {line_number}.")

            if not isinstance(text, str):
                raise ValueError(f"Invalid text for record {record_id}.")

            if not isinstance(label, str):
                raise ValueError(f"Invalid label for record {record_id}.")

            if record_id in records:
                raise ValueError(f"Duplicate record ID: {record_id}")

            records[record_id] = {
                "text": text,
                "label": label,
            }

    return records


def build_vocabulary(
    training_texts: list[str],
    *,
    max_size: int,
    min_frequency: int,
) -> dict[str, int]:
    """Build a deterministic vocabulary using training documents only."""
    if max_size < 3:
        raise ValueError("max_size must be at least 3.")

    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1.")

    token_counts: Counter[str] = Counter()

    for text in training_texts:
        token_counts.update(tokenize_words(text))

    eligible_tokens = [
        (token, count)
        for token, count in token_counts.items()
        if count >= min_frequency and token not in {PAD_TOKEN, UNKNOWN_TOKEN}
    ]

    eligible_tokens.sort(key=lambda item: (-item[1], item[0]))

    selected_tokens = eligible_tokens[: max_size - 2]

    vocabulary = {
        PAD_TOKEN: PADDING_INDEX,
        UNKNOWN_TOKEN: UNKNOWN_INDEX,
    }

    for token, _ in selected_tokens:
        vocabulary[token] = len(vocabulary)

    return vocabulary


def encode_documents(
    documents: list[str],
    vocabulary: dict[str, int],
    *,
    max_sequence_length: int,
) -> torch.Tensor:
    """Convert documents to padded, truncated integer sequences."""
    if max_sequence_length < 1:
        raise ValueError("max_sequence_length must be positive.")

    encoded = torch.full(
        (len(documents), max_sequence_length),
        fill_value=PADDING_INDEX,
        dtype=torch.long,
    )

    for row_index, document in enumerate(documents):
        token_ids = [
            vocabulary.get(token, UNKNOWN_INDEX) for token in tokenize_words(document)
        ][:max_sequence_length]

        if token_ids:
            encoded[row_index, : len(token_ids)] = torch.tensor(
                token_ids,
                dtype=torch.long,
            )

    return encoded


def _prepare_documents(
    records: list[dict[str, str]],
    *,
    representation: str,
    lowercase: bool,
) -> list[str]:
    return [
        preprocess_text(
            record["text"],
            representation=representation,
            lowercase=lowercase,
        )
        for record in records
    ]


def _predict(
    model: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, ...]],
    *,
    device: torch.device,
) -> tuple[list[int], float]:
    """Generate predictions and measure elapsed inference time."""
    model.eval()
    predictions: list[int] = []

    prediction_start = time.perf_counter()

    with torch.inference_mode():
        for input_ids, _ in data_loader:
            input_ids = input_ids.to(device)
            logits = model(input_ids)
            batch_predictions = torch.argmax(logits, dim=1)
            predictions.extend(batch_predictions.cpu().tolist())

    prediction_seconds = time.perf_counter() - prediction_start

    return predictions, prediction_seconds


def _write_metrics_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id",
        "experiment_id",
        "model",
        "representation",
        "training_documents",
        "validation_documents",
        "vocabulary_size",
        "best_epoch",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "fit_seconds",
        "prediction_seconds",
        "latency_ms_per_document",
        "throughput_documents_per_second",
        "artifact_size_bytes",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def run_neural_experiments(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Train and validate a configured neural model without using test records."""
    data_config = config["data"]
    preprocessing_config = config["preprocessing"]
    vocabulary_config = config["vocabulary"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["outputs"]

    raw_path = project_root / data_config["raw_dataset_path"]
    split_path = project_root / data_config["split_manifest_path"]

    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    random_seed = int(config["project"]["random_seed"])

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    num_threads = int(training_config.get("num_threads", 1))

    if num_threads < 1:
        raise ValueError("num_threads must be positive.")

    torch.set_num_threads(num_threads)

    device = torch.device("cpu")

    records_by_id = _load_training_records(raw_path)
    split = _read_json(split_path)

    training_ids = split.get("training_record_ids")
    validation_ids = split.get("validation_record_ids")

    if not isinstance(training_ids, list) or not isinstance(validation_ids, list):
        raise ValueError("Split manifest must contain training and validation IDs.")

    if not all(isinstance(value, str) for value in training_ids + validation_ids):
        raise ValueError("All split record IDs must be strings.")

    training_id_set = set(training_ids)
    validation_id_set = set(validation_ids)

    if not training_id_set or not validation_id_set:
        raise ValueError("Training and validation partitions cannot be empty.")

    if not training_id_set.isdisjoint(validation_id_set):
        raise ValueError("Training and validation record IDs overlap.")

    unknown_ids = (training_id_set | validation_id_set) - set(records_by_id)

    if unknown_ids:
        raise ValueError(
            f"The split manifest contains {len(unknown_ids)} "
            "IDs not found in the official training partition."
        )

    training_records = [records_by_id[value] for value in training_ids]
    validation_records = [records_by_id[value] for value in validation_ids]

    class_labels = sorted({record["label"] for record in training_records})

    if {record["label"] for record in validation_records} != set(class_labels):
        raise ValueError("Training and validation must contain the same class labels.")

    label_to_index = {label: index for index, label in enumerate(class_labels)}

    y_train = torch.tensor(
        [label_to_index[record["label"]] for record in training_records],
        dtype=torch.long,
    )

    y_validation = torch.tensor(
        [label_to_index[record["label"]] for record in validation_records],
        dtype=torch.long,
    )

    representation = str(preprocessing_config["representation"])
    lowercase = bool(preprocessing_config.get("lowercase", False))

    X_train = _prepare_documents(
        training_records,
        representation=representation,
        lowercase=lowercase,
    )
    X_validation = _prepare_documents(
        validation_records,
        representation=representation,
        lowercase=lowercase,
    )

    vocabulary = build_vocabulary(
        X_train,
        max_size=int(vocabulary_config["max_size"]),
        min_frequency=int(vocabulary_config["min_frequency"]),
    )

    max_sequence_length = int(training_config["max_sequence_length"])
    architecture = str(model_config.get("architecture", "textcnn"))

    if architecture == "textcnn":
        kernel_sizes = tuple(int(value) for value in model_config["kernel_sizes"])

        if max_sequence_length < max(kernel_sizes):
            raise ValueError(
                "max_sequence_length must be at least the largest kernel size."
            )

    X_train_encoded = encode_documents(
        X_train,
        vocabulary,
        max_sequence_length=max_sequence_length,
    )
    X_validation_encoded = encode_documents(
        X_validation,
        vocabulary,
        max_sequence_length=max_sequence_length,
    )

    training_dataset = TensorDataset(X_train_encoded, y_train)
    validation_dataset = TensorDataset(X_validation_encoded, y_validation)

    batch_size = int(training_config["batch_size"])

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    generator = torch.Generator()
    generator.manual_seed(random_seed)

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    if architecture == "textcnn":
        model: nn.Module = TextCNN(
            vocabulary_size=len(vocabulary),
            num_classes=len(class_labels),
            embedding_dim=int(model_config["embedding_dim"]),
            num_filters=int(model_config["num_filters"]),
            kernel_sizes=kernel_sizes,
            dropout=float(model_config["dropout"]),
            padding_index=PADDING_INDEX,
        )

        experiment_id = "EXP-04"
        model_name = "TextCNN"

    elif architecture == "bilstm":
        model = BiLSTM(
            vocabulary_size=len(vocabulary),
            num_classes=len(class_labels),
            embedding_dim=int(model_config["embedding_dim"]),
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
            dropout=float(model_config["dropout"]),
            padding_index=PADDING_INDEX,
        )

        experiment_id = "EXP-05"
        model_name = "BiLSTM"

    else:
        raise ValueError(f"Unsupported neural architecture: {architecture!r}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    epochs = int(training_config["epochs"])
    patience = int(training_config["patience"])
    max_grad_norm = float(training_config["max_grad_norm"])

    if epochs < 1 or patience < 1 or max_grad_norm <= 0:
        raise ValueError("epochs, patience and max_grad_norm must be positive.")

    best_macro_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    fit_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        documents_seen = 0

        for input_ids, labels in training_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )

            optimizer.step()

            batch_count = labels.size(0)
            total_loss += loss.item() * batch_count
            documents_seen += batch_count

        epoch_loss = total_loss / documents_seen

        validation_predictions, _ = _predict(
            model,
            validation_loader,
            device=device,
        )

        validation_metrics = calculate_classification_metrics(
            [str(value) for value in y_validation.tolist()],
            [str(value) for value in validation_predictions],
            class_labels=[str(index) for index in range(len(class_labels))],
        )

        epoch_result: dict[str, float | int] = {
            "epoch": epoch,
            "training_loss": epoch_loss,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"loss={epoch_loss:.4f} | "
            f"validation_macro_f1={validation_metrics['macro_f1']:.4f}"
        )

        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                "Early stopping: validation macro F1 did not improve "
                f"for {patience} consecutive epochs."
            )
            break

    fit_seconds = time.perf_counter() - fit_start

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")

    model.load_state_dict(best_state)

    predicted_indices, prediction_seconds = _predict(
        model,
        validation_loader,
        device=device,
    )

    y_true = [class_labels[index] for index in y_validation.tolist()]
    y_pred = [class_labels[index] for index in predicted_indices]

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        class_labels=class_labels,
    )

    model_payload = {
        "model_state_dict": best_state,
        "vocabulary": vocabulary,
        "class_labels": class_labels,
        "model_config": {
            "architecture": architecture,
            **model_config,
            "padding_index": PADDING_INDEX,
        },
        "max_sequence_length": max_sequence_length,
        "representation": representation,
        "lowercase": lowercase,
        "random_seed": random_seed,
    }

    artifact_dir = project_root / training_config["artifact_dir"]
    run_id = f"{architecture}__{representation}"

    if training_config.get("save_model", False):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{run_id}.pt"
        torch.save(model_payload, artifact_path)
        artifact_size_bytes = artifact_path.stat().st_size
        saved_artifact = str(artifact_path.relative_to(project_root))
    else:
        with tempfile.TemporaryDirectory() as temporary_dir:
            artifact_path = Path(temporary_dir) / "model.pt"
            torch.save(model_payload, artifact_path)
            artifact_size_bytes = artifact_path.stat().st_size
        saved_artifact = None

    validation_count = len(y_validation)
    latency_ms = prediction_seconds / validation_count * 1000
    throughput = (
        validation_count / prediction_seconds if prediction_seconds > 0 else None
    )

    result = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "model": model_name,
        "representation": representation,
        "random_seed": random_seed,
        "training_documents": len(y_train),
        "validation_documents": validation_count,
        "vocabulary_size": len(vocabulary),
        "max_sequence_length": max_sequence_length,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "latency_ms_per_document": latency_ms,
        "throughput_documents_per_second": throughput,
        "artifact_size_bytes": artifact_size_bytes,
        "saved_artifact": saved_artifact,
        "metrics": metrics,
        "history": history,
        "model_parameters": model_config,
        "training_parameters": training_config,
        "test_data_used": False,
    }

    summary_row = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "model": model_name,
        "representation": representation,
        "training_documents": len(y_train),
        "validation_documents": validation_count,
        "vocabulary_size": len(vocabulary),
        "best_epoch": best_epoch,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "latency_ms_per_document": latency_ms,
        "throughput_documents_per_second": throughput,
        "artifact_size_bytes": artifact_size_bytes,
    }

    metrics_path = project_root / output_config["metrics_csv"]
    results_path = project_root / output_config["results_json"]

    _write_metrics_csv(metrics_path, [summary_row])

    results_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": "20newsgroups",
        "development_split": str(data_config["split_manifest_path"]),
        "random_seed": random_seed,
        "training_documents": len(y_train),
        "validation_documents": validation_count,
        "class_labels": class_labels,
        "primary_selection_metric": "macro_f1",
        "test_data_used": False,
        "runs": [result],
    }

    results_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "runs": [result],
        "summary": [summary_row],
        "metrics_csv": str(metrics_path),
        "results_json": str(results_path),
    }
