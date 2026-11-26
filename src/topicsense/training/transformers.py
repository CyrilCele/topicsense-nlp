"""Reproducible transformer experiments for TopicSense-NLP."""

from __future__ import annotations

import csv
import inspect
import json
import random
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import psutil
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from topicsense.evaluation.metrics import calculate_classification_metrics
from topicsense.preprocessing.text import preprocess_text


class ProcessMemoryMonitor:
    """Sample process RSS periodically and retain the highest observed value."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        if interval_seconds <= 0:
            raise ValueError("Memory sampling interval must be positive.")

        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes: int = int(self.process.memory_info().rss)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> None:
        """Update the highest sampled resident memory usage."""
        current_rss = int(self.process.memory_info().rss)
        self.peak_rss_bytes = max(self.peak_rss_bytes, current_rss)

    def _monitor(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.sample()
            except psutil.Error:
                break

    def start(self) -> None:
        """Start periodic memory sampling."""
        if self._thread is not None:
            return

        self.sample()
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._monitor,
            name="topicsense-memory-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and record a final memory measurement."""
        if self._thread is None:
            self.sample()
            return

        self._stop_event.set()
        self._thread.join()
        self._thread = None
        self.sample()


class MemoryMonitorCallback(TrainerCallback):
    """Record sampled process memory during model fine-tuning."""

    def __init__(self) -> None:
        self.monitor = ProcessMemoryMonitor()

    @property
    def peak_rss_bytes(self) -> int:
        return self.monitor.peak_rss_bytes

    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        self.monitor.start()
        return control

    def on_train_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        self.monitor.stop()
        return control


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

            records[record_id] = {"text": text, "label": label}

    return records


def _load_development_data(
    *,
    raw_path: Path,
    split_path: Path,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load the fixed training and validation partitions without test records."""
    records_by_id = _load_training_records(raw_path)
    split = _read_json(split_path)

    training_ids = split.get("training_record_ids")
    validation_ids = split.get("validation_record_ids")

    if not isinstance(training_ids, list) or not isinstance(validation_ids, list):
        raise ValueError("Split manifest must contain training and validation IDs.")

    if not all(
        isinstance(record_id, str) for record_id in training_ids + validation_ids
    ):
        raise ValueError("All split record IDs must be strings.")

    training_id_set = set(training_ids)
    validation_id_set = set(validation_ids)

    if not training_id_set or not validation_id_set:
        raise ValueError("Training and validation partitions cannot be empty.")

    if len(training_id_set) != len(training_ids):
        raise ValueError("Training IDs contain duplicates.")

    if len(validation_id_set) != len(validation_ids):
        raise ValueError("Validation IDs contain duplicates.")

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

    training_labels = [record["label"] for record in training_records]
    validation_labels = [record["label"] for record in validation_records]

    class_labels = sorted(set(training_labels))

    if set(validation_labels) != set(class_labels):
        raise ValueError("Training and validation must contain the same classes.")

    return (
        [record["text"] for record in training_records],
        training_labels,
        [record["text"] for record in validation_records],
        validation_labels,
    )


def _prepare_texts(
    texts: list[str],
    *,
    representation: str,
    lowercase: bool,
) -> list[str]:
    return [
        preprocess_text(
            text,
            representation=representation,
            lowercase=lowercase,
        )
        for text in texts
    ]


def _write_result(
    *,
    metrics_path: Path,
    results_path: Path,
    summary: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    results_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _prepare_artifact_directory(
    *,
    project_root: Path,
    relative_path: str,
    save_model: bool,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if save_model:
        artifact_path = project_root / relative_path
        artifact_path.mkdir(parents=True, exist_ok=True)
        return artifact_path, None

    temporary_directory = tempfile.TemporaryDirectory()
    return Path(temporary_directory.name) / "artifact", temporary_directory


def _compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_labels: list[str],
) -> dict[str, Any]:
    predicted_indices = np.argmax(predictions, axis=1)

    return calculate_classification_metrics(
        [class_labels[int(value)] for value in labels],
        [class_labels[int(value)] for value in predicted_indices],
        class_labels=class_labels,
    )


def _tokenize_dataset(
    texts: list[str],
    labels: list[int],
    *,
    tokenizer: Any,
    max_length: int,
    description: str,
) -> Dataset:
    """Tokenize documents and return a labelled Hugging Face dataset."""

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return cast(dict[str, Any], tokenized)

    dataset = Dataset.from_dict({"text": texts, "label": labels})

    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc=description,
    )

    return dataset.rename_column("label", "labels")


def _best_checkpoint_information(
    trainer: Trainer,
) -> tuple[str | None, float | None, float | None]:
    """Return the best checkpoint name, validation macro F1 and its epoch."""
    best_metric = trainer.state.best_metric
    checkpoint_path = trainer.state.best_model_checkpoint

    checkpoint_name = (
        Path(checkpoint_path).name if checkpoint_path is not None else None
    )

    best_epoch: float | None = None
    best_validation_macro_f1: float | None = (
        float(best_metric) if best_metric is not None else None
    )

    if best_validation_macro_f1 is not None:
        matching_evaluations = [
            entry
            for entry in trainer.state.log_history
            if "eval_macro_f1" in entry
            and "epoch" in entry
            and np.isclose(
                float(entry["eval_macro_f1"]),
                best_validation_macro_f1,
            )
        ]

        if matching_evaluations:
            best_epoch = float(matching_evaluations[-1]["epoch"])

    return checkpoint_name, best_validation_macro_f1, best_epoch


def run_distilbert_experiment(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Fine-tune DistilBERT using the fixed development split."""
    settings = config["fine_tuning"]

    if not settings.get("enabled", True):
        raise ValueError("DistilBERT experiment is disabled in the configuration.")

    random_seed = int(config["project"]["random_seed"])
    set_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    raw_path = project_root / config["data"]["raw_dataset_path"]
    split_path = project_root / config["data"]["split_manifest_path"]

    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    train_texts, train_labels, validation_texts, validation_labels = (
        _load_development_data(
            raw_path=raw_path,
            split_path=split_path,
        )
    )

    class_labels = sorted(set(train_labels))
    label_to_id = {label: index for index, label in enumerate(class_labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    representation = str(config["preprocessing"]["representation"])
    lowercase = bool(config["preprocessing"].get("lowercase", False))

    train_texts = _prepare_texts(
        train_texts,
        representation=representation,
        lowercase=lowercase,
    )
    validation_texts = _prepare_texts(
        validation_texts,
        representation=representation,
        lowercase=lowercase,
    )

    train_label_ids = [label_to_id[label] for label in train_labels]
    validation_label_ids = [label_to_id[label] for label in validation_labels]

    model_name = str(settings["model_name"])
    max_length = int(settings["max_sequence_length"])

    if max_length < 8:
        raise ValueError("max_sequence_length must be at least 8.")

    if int(settings["batch_size"]) < 1:
        raise ValueError("DistilBERT batch size must be positive.")

    if int(settings["gradient_accumulation_steps"]) < 1:
        raise ValueError("Gradient accumulation steps must be positive.")

    if int(settings["epochs"]) < 1:
        raise ValueError("DistilBERT epochs must be positive.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = _tokenize_dataset(
        train_texts,
        train_label_ids,
        tokenizer=tokenizer,
        max_length=max_length,
        description="Tokenising training documents",
    )
    validation_dataset = _tokenize_dataset(
        validation_texts,
        validation_label_ids,
        tokenizer=tokenizer,
        max_length=max_length,
        description="Tokenising validation documents",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(class_labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    model.config.problem_type = "single_label_classification"

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(evaluation: Any) -> dict[str, float]:
        logits, labels = evaluation
        metrics = _compute_metrics(
            np.asarray(logits),
            np.asarray(labels),
            class_labels,
        )

        return {
            "accuracy": float(metrics["accuracy"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
        }

    with tempfile.TemporaryDirectory() as temporary_directory:
        trainer_output_dir = Path(temporary_directory) / "trainer"

        argument_values: dict[str, Any] = {
            "output_dir": str(trainer_output_dir),
            "num_train_epochs": int(settings["epochs"]),
            "per_device_train_batch_size": int(settings["batch_size"]),
            "per_device_eval_batch_size": int(settings["batch_size"]),
            "gradient_accumulation_steps": int(settings["gradient_accumulation_steps"]),
            "learning_rate": float(settings["learning_rate"]),
            "weight_decay": float(settings["weight_decay"]),
            "logging_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "macro_f1",
            "greater_is_better": True,
            "save_total_limit": 1,
            "report_to": "none",
            "seed": random_seed,
            "data_seed": random_seed,
            "dataloader_num_workers": int(settings.get("num_workers", 0)),
            "dataloader_pin_memory": False,
            "remove_unused_columns": True,
        }

        training_argument_parameters = inspect.signature(TrainingArguments).parameters

        warmup_ratio = float(settings["warmup_ratio"])

        if not 0.0 <= warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be between 0 and 1.")

        if "warmup_ratio" in training_argument_parameters:
            argument_values["warmup_ratio"] = warmup_ratio

        elif "warmup_steps" in training_argument_parameters:
            batch_size = int(settings["batch_size"])
            gradient_accumulation_steps = int(settings["gradient_accumulation_steps"])
            epochs = int(settings["epochs"])

            batches_per_epoch = (len(train_dataset) + batch_size - 1) // batch_size

            optimizer_steps_per_epoch = max(
                (batches_per_epoch + gradient_accumulation_steps - 1)
                // gradient_accumulation_steps,
                1,
            )

            total_training_steps = optimizer_steps_per_epoch * epochs

            argument_values["warmup_steps"] = int(
                np.ceil(total_training_steps * warmup_ratio)
            )

        else:
            raise RuntimeError(
                "The installed Transformers version does not expose "
                "a supported warm-up configuration argument."
            )

        if "eval_strategy" in training_argument_parameters:
            argument_values["eval_strategy"] = "epoch"
        elif "evaluation_strategy" in training_argument_parameters:
            argument_values["evaluation_strategy"] = "epoch"
        else:
            raise RuntimeError(
                "The installed Transformers version does not expose "
                "a supported evaluation strategy argument."
            )

        if "use_cpu" in training_argument_parameters:
            argument_values["use_cpu"] = True
        elif "no_cuda" in training_argument_parameters:
            argument_values["no_cuda"] = True
        else:
            raise RuntimeError(
                "The installed Transformers version does not expose "
                "a supported CPU-only execution argument."
            )

        training_arguments = TrainingArguments(**argument_values)

        memory_monitor = MemoryMonitorCallback()

        trainer_parameters = inspect.signature(Trainer).parameters

        trainer_values: dict[str, Any] = {
            "model": model,
            "args": training_arguments,
            "train_dataset": train_dataset,
            "eval_dataset": validation_dataset,
            "data_collator": data_collator,
            "compute_metrics": compute_metrics,
            "callbacks": [
                EarlyStoppingCallback(
                    early_stopping_patience=int(settings["patience"])
                ),
                memory_monitor,
            ],
        }

        if "processing_class" in trainer_parameters:
            trainer_values["processing_class"] = tokenizer
        else:
            trainer_values["tokenizer"] = tokenizer

        trainer = Trainer(**trainer_values)

        fit_start = time.perf_counter()
        training_output = trainer.train()
        fit_seconds = time.perf_counter() - fit_start

        # Measure validation tokenisation separately so that the complete
        # inference pipeline can be reported independently of prediction.
        tokenization_start = time.perf_counter()

        inference_dataset = _tokenize_dataset(
            validation_texts,
            validation_label_ids,
            tokenizer=tokenizer,
            max_length=max_length,
            description="Preparing validation inference documents",
        )

        validation_tokenization_seconds = time.perf_counter() - tokenization_start

        prediction_start = time.perf_counter()
        prediction_output = trainer.predict(inference_dataset)
        prediction_seconds = time.perf_counter() - prediction_start

        end_to_end_inference_seconds = (
            validation_tokenization_seconds + prediction_seconds
        )

        prediction_logits = np.asarray(prediction_output.predictions)
        prediction_labels = np.asarray(prediction_output.label_ids)

        metrics = _compute_metrics(
            prediction_logits,
            prediction_labels,
            class_labels,
        )

        (
            best_checkpoint,
            best_validation_macro_f1,
            best_epoch,
        ) = _best_checkpoint_information(trainer)

        artifact_path, artifact_temporary_directory = _prepare_artifact_directory(
            project_root=project_root,
            relative_path=str(settings["artifact_dir"]),
            save_model=bool(settings.get("save_model", False)),
        )

        try:
            trainer.save_model(str(artifact_path))
            tokenizer.save_pretrained(str(artifact_path))
            model_and_tokenizer_artifact_size_bytes = _directory_size(artifact_path)

            saved_artifact = (
                str(artifact_path.relative_to(project_root))
                if settings.get("save_model", False)
                else None
            )
        finally:
            if artifact_temporary_directory is not None:
                artifact_temporary_directory.cleanup()

        epochs_completed = (
            float(trainer.state.epoch) if trainer.state.epoch is not None else 0.0
        )

    summary = {
        "run_id": "distilbert__message_body",
        "experiment_id": "EXP-06",
        "model": model_name,
        "training_documents": len(train_texts),
        "validation_documents": len(validation_texts),
        "max_sequence_length": max_length,
        "batch_size": int(settings["batch_size"]),
        "gradient_accumulation_steps": int(settings["gradient_accumulation_steps"]),
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_validation_macro_f1,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "fit_seconds": fit_seconds,
        "validation_tokenization_seconds": validation_tokenization_seconds,
        "prediction_seconds": prediction_seconds,
        "end_to_end_inference_seconds": end_to_end_inference_seconds,
        "prediction_latency_ms_per_document": (
            prediction_seconds / len(validation_texts) * 1000
        ),
        "end_to_end_latency_ms_per_document": (
            end_to_end_inference_seconds / len(validation_texts) * 1000
        ),
        "end_to_end_throughput_documents_per_second": (
            len(validation_texts) / end_to_end_inference_seconds
            if end_to_end_inference_seconds > 0
            else None
        ),
        "model_and_tokenizer_artifact_size_bytes": (
            model_and_tokenizer_artifact_size_bytes
        ),
        "artifact_size_scope": "Fine-tuned model and tokenizer",
        "peak_sampled_rss_bytes": memory_monitor.peak_rss_bytes,
    }

    payload = {
        "dataset": "20newsgroups",
        "development_split": str(config["data"]["split_manifest_path"]),
        "random_seed": random_seed,
        "primary_selection_metric": "macro_f1",
        "test_data_used": False,
        "best_checkpoint_name": best_checkpoint,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_validation_macro_f1,
        "training_metrics": training_output.metrics,
        "model_parameters": settings,
        "preprocessing": config["preprocessing"],
        "class_labels": class_labels,
        "summary": summary,
        "metrics": metrics,
        "saved_artifact": saved_artifact,
    }

    _write_result(
        metrics_path=project_root / config["outputs"]["distilbert_metrics_csv"],
        results_path=project_root / config["outputs"]["distilbert_results_json"],
        summary=summary,
        payload=payload,
    )

    return {"summary": summary, "metrics": metrics, "results": payload}


def _encode_documents(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
    normalize_embeddings: bool,
    description: str,
    memory_monitor: ProcessMemoryMonitor | None = None,
) -> np.ndarray:
    """Encode documents in bounded batches with visible progress."""
    if batch_size < 1:
        raise ValueError("Embedding batch size must be positive.")

    batches: list[np.ndarray] = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc=description,
        unit="batch",
    ):
        batch_texts = texts[start : start + batch_size]

        embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )

        batches.append(np.asarray(embeddings, dtype=np.float32))

        if memory_monitor is not None:
            memory_monitor.sample()

    if not batches:
        raise ValueError("Cannot generate embeddings for an empty document list.")

    return np.concatenate(batches, axis=0)


def run_frozen_embedding_experiment(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Train a lightweight classifier on frozen sentence-transformer embeddings."""
    settings = config["frozen_embeddings"]

    if not settings.get("enabled", True):
        raise ValueError("Frozen embedding experiment is disabled.")

    random_seed = int(config["project"]["random_seed"])
    random.seed(random_seed)
    np.random.seed(random_seed)

    raw_path = project_root / config["data"]["raw_dataset_path"]
    split_path = project_root / config["data"]["split_manifest_path"]

    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    train_texts, train_labels, validation_texts, validation_labels = (
        _load_development_data(
            raw_path=raw_path,
            split_path=split_path,
        )
    )

    class_labels = sorted(set(train_labels))
    representation = str(config["preprocessing"]["representation"])
    lowercase = bool(config["preprocessing"].get("lowercase", False))

    train_texts = _prepare_texts(
        train_texts,
        representation=representation,
        lowercase=lowercase,
    )
    validation_texts = _prepare_texts(
        validation_texts,
        representation=representation,
        lowercase=lowercase,
    )

    batch_size = int(settings["batch_size"])

    if batch_size < 1:
        raise ValueError("Frozen embedding batch size must be positive.")

    classifier_settings = settings["classifier"]

    if int(classifier_settings["max_iter"]) < 1:
        raise ValueError("Classifier max_iter must be positive.")

    if float(classifier_settings["C"]) <= 0:
        raise ValueError("Classifier C must be positive.")

    model_name = str(settings["model_name"])
    normalize_embeddings = bool(settings["normalize_embeddings"])

    memory_monitor = ProcessMemoryMonitor()
    memory_monitor.start()

    try:
        encoder = SentenceTransformer(model_name, device="cpu")

        training_embedding_start = time.perf_counter()

        training_embeddings = _encode_documents(
            encoder,
            train_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            description="Encoding training documents",
            memory_monitor=memory_monitor,
        )

        training_embedding_seconds = time.perf_counter() - training_embedding_start

        classifier = LogisticRegression(
            C=float(classifier_settings["C"]),
            max_iter=int(classifier_settings["max_iter"]),
            random_state=random_seed,
        )

        fit_start = time.perf_counter()
        classifier.fit(training_embeddings, train_labels)
        fit_seconds = time.perf_counter() - fit_start

        memory_monitor.sample()

        # Measure the complete validation inference pipeline. This includes
        # generating fresh embeddings and obtaining classifier predictions.
        inference_embedding_start = time.perf_counter()

        inference_embeddings = _encode_documents(
            encoder,
            validation_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            description="Encoding validation inference documents",
            memory_monitor=memory_monitor,
        )

        inference_embedding_seconds = time.perf_counter() - inference_embedding_start

        prediction_start = time.perf_counter()
        predictions = classifier.predict(inference_embeddings)
        prediction_seconds = time.perf_counter() - prediction_start

        memory_monitor.sample()

        end_to_end_inference_seconds = inference_embedding_seconds + prediction_seconds

        metrics = calculate_classification_metrics(
            validation_labels,
            [str(value) for value in predictions],
            class_labels=class_labels,
        )

        artifact_path, artifact_temporary_directory = _prepare_artifact_directory(
            project_root=project_root,
            relative_path=str(settings["artifact_dir"]),
            save_model=bool(settings.get("save_model", False)),
        )

        try:
            artifact_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(classifier, artifact_path / "classifier.joblib")

            metadata = {
                "experiment_id": "EXP-07",
                "embedding_model": model_name,
                "embedding_model_is_frozen": True,
                "encoder_artifact_included": False,
                "class_labels": class_labels,
                "representation": representation,
                "lowercase": lowercase,
                "normalize_embeddings": normalize_embeddings,
                "random_seed": random_seed,
                "classifier_parameters": classifier_settings,
            }

            (artifact_path / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            classifier_artifact_size_bytes = _directory_size(artifact_path)

            saved_artifact = (
                str(artifact_path.relative_to(project_root))
                if settings.get("save_model", False)
                else None
            )
        finally:
            if artifact_temporary_directory is not None:
                artifact_temporary_directory.cleanup()

        peak_sampled_rss_bytes = memory_monitor.peak_rss_bytes

    finally:
        memory_monitor.stop()

    summary = {
        "run_id": "frozen_minilm__message_body",
        "experiment_id": "EXP-07",
        "model": model_name,
        "classifier": "LogisticRegression",
        "training_documents": len(train_texts),
        "validation_documents": len(validation_texts),
        "embedding_dimension": int(training_embeddings.shape[1]),
        "training_embedding_seconds": training_embedding_seconds,
        "fit_seconds": fit_seconds,
        "inference_embedding_seconds": inference_embedding_seconds,
        "prediction_seconds": prediction_seconds,
        "end_to_end_inference_seconds": end_to_end_inference_seconds,
        "prediction_latency_ms_per_document": (
            prediction_seconds / len(validation_texts) * 1000
        ),
        "end_to_end_latency_ms_per_document": (
            end_to_end_inference_seconds / len(validation_texts) * 1000
        ),
        "end_to_end_throughput_documents_per_second": (
            len(validation_texts) / end_to_end_inference_seconds
            if end_to_end_inference_seconds > 0
            else None
        ),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "classifier_artifact_size_bytes": classifier_artifact_size_bytes,
        "artifact_size_scope": (
            "Logistic Regression classifier and metadata; "
            "pretrained SentenceTransformer encoder excluded"
        ),
        "peak_sampled_rss_bytes": peak_sampled_rss_bytes,
    }

    payload = {
        "dataset": "20newsgroups",
        "development_split": str(config["data"]["split_manifest_path"]),
        "random_seed": random_seed,
        "primary_selection_metric": "macro_f1",
        "test_data_used": False,
        "model_parameters": settings,
        "preprocessing": config["preprocessing"],
        "class_labels": class_labels,
        "embedding_model": {
            "name": model_name,
            "frozen": True,
            "artifact_included": False,
        },
        "summary": summary,
        "metrics": metrics,
        "saved_artifact": saved_artifact,
    }

    _write_result(
        metrics_path=project_root / config["outputs"]["frozen_metrics_csv"],
        results_path=project_root / config["outputs"]["frozen_results_json"],
        summary=summary,
        payload=payload,
    )

    return {"summary": summary, "metrics": metrics, "results": payload}


def run_transformer_experiments(
    config: dict[str, Any],
    *,
    project_root: Path,
    experiment: str = "all",
) -> dict[str, Any]:
    """Run one or both configured transformer experiments."""
    if experiment not in {"all", "distilbert", "frozen_embeddings"}:
        raise ValueError(
            "experiment must be 'all', 'distilbert' or 'frozen_embeddings'."
        )

    results: dict[str, Any] = {}

    if experiment in {"all", "distilbert"}:
        results["distilbert"] = run_distilbert_experiment(
            config,
            project_root=project_root,
        )

    if experiment in {"all", "frozen_embeddings"}:
        results["frozen_embeddings"] = run_frozen_embedding_experiment(
            config,
            project_root=project_root,
        )

    return results
