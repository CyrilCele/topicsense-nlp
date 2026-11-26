"""Reproducible classical machine-learning baseline experiments."""

from __future__ import annotations

import csv
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from topicsense.evaluation.metrics import calculate_classification_metrics
from topicsense.features.vectorization import build_vectorizer
from topicsense.models.classical.estimators import build_classifier
from topicsense.preprocessing.text import preprocess_text


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")

    return payload


def _load_training_records(path: Path) -> dict[str, dict[str, Any]]:
    """Load official training records only; never retain official test records."""
    records: dict[str, dict[str, Any]] = {}

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


def _write_metrics_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id",
        "experiment_id",
        "classifier",
        "feature_type",
        "representation",
        "training_documents",
        "validation_documents",
        "feature_count",
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


def run_classical_experiments(
    config: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Train and evaluate configured classical baselines on a fixed split.

    The input dataset is filtered to official training records. The supplied
    split manifest determines the training and validation IDs. Official test
    records are never passed to a vectorizer, estimator or evaluator.
    """
    data_config = config["data"]
    preprocessing_config = config["preprocessing"]
    feature_config = config["features"]
    classifier_config = config["classifiers"]
    training_config = config["training"]
    output_config = config["outputs"]

    raw_path = project_root / data_config["raw_dataset_path"]
    split_path = project_root / data_config["split_manifest_path"]

    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    if not split_path.is_file():
        raise FileNotFoundError(split_path)

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

    y_train = [record["label"] for record in training_records]
    y_validation = [record["label"] for record in validation_records]

    class_labels = sorted(set(y_train))

    if set(y_validation) != set(class_labels):
        raise ValueError("Training and validation must contain the same class labels.")

    random_seed = int(config["project"]["random_seed"])
    lowercase = bool(preprocessing_config.get("lowercase", False))

    feature_types = feature_config["types"]
    representations = preprocessing_config["representations"]

    enabled_classifiers = {
        name: settings
        for name, settings in classifier_config.items()
        if settings.get("enabled", False)
    }

    if not enabled_classifiers:
        raise ValueError("At least one classifier must be enabled.")

    results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for representation in representations:
        X_train = [
            preprocess_text(
                record["text"],
                representation=representation,
                lowercase=lowercase,
            )
            for record in training_records
        ]

        X_validation = [
            preprocess_text(
                record["text"],
                representation=representation,
                lowercase=lowercase,
            )
            for record in validation_records
        ]

        for feature_type in feature_types:
            for classifier_name, classifier_settings in enabled_classifiers.items():
                experiment_ids = {
                    "multinomial_nb": "EXP-01",
                    "logistic_regression": "EXP-02",
                    "linear_svm": "EXP-03",
                }

                experiment_id = experiment_ids[classifier_name]
                run_id = f"{classifier_name}__{feature_type}__{representation}"

                vectorizer = build_vectorizer(feature_type, feature_config)
                classifier = build_classifier(
                    classifier_name,
                    classifier_settings,
                    random_seed=random_seed,
                )

                pipeline = Pipeline(
                    steps=[
                        ("vectorizer", vectorizer),
                        ("classifier", classifier),
                    ]
                )

                fit_start = time.perf_counter()
                pipeline.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - fit_start

                prediction_start = time.perf_counter()
                predictions = pipeline.predict(X_validation)
                prediction_seconds = time.perf_counter() - prediction_start

                y_pred = [str(value) for value in predictions]

                metrics = calculate_classification_metrics(
                    y_validation,
                    y_pred,
                    class_labels=class_labels,
                )

                feature_count = len(
                    pipeline.named_steps["vectorizer"].get_feature_names_out()
                )

                artifact_dir = project_root / training_config["artifact_dir"]

                if training_config.get("save_models", False):
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    artifact_path = artifact_dir / f"{run_id}.joblib"
                    joblib.dump(pipeline, artifact_path)
                    artifact_size_bytes = artifact_path.stat().st_size
                    saved_artifact = str(artifact_path.relative_to(project_root))
                else:
                    with tempfile.TemporaryDirectory() as temporary_dir:
                        artifact_path = Path(temporary_dir) / "model.joblib"
                        joblib.dump(pipeline, artifact_path)
                        artifact_size_bytes = artifact_path.stat().st_size
                    saved_artifact = None

                validation_count = len(y_validation)

                result = {
                    "run_id": run_id,
                    "experiment_id": experiment_id,
                    "classifier": classifier_name,
                    "feature_type": feature_type,
                    "representation": representation,
                    "random_seed": random_seed,
                    "training_documents": len(y_train),
                    "validation_documents": validation_count,
                    "feature_count": feature_count,
                    "fit_seconds": fit_seconds,
                    "prediction_seconds": prediction_seconds,
                    "latency_ms_per_document": (
                        prediction_seconds / validation_count * 1000
                    ),
                    "throughput_documents_per_second": (
                        validation_count / prediction_seconds
                        if prediction_seconds > 0
                        else None
                    ),
                    "artifact_size_bytes": artifact_size_bytes,
                    "saved_artifact": saved_artifact,
                    "metrics": metrics,
                    "classifier_parameters": classifier_settings,
                    "feature_parameters": feature_config,
                }

                results.append(result)

                summary_rows.append(
                    {
                        "run_id": run_id,
                        "experiment_id": experiment_id,
                        "classifier": classifier_name,
                        "feature_type": feature_type,
                        "representation": representation,
                        "training_documents": len(y_train),
                        "validation_documents": validation_count,
                        "feature_count": feature_count,
                        "accuracy": metrics["accuracy"],
                        "macro_precision": metrics["macro_precision"],
                        "macro_recall": metrics["macro_recall"],
                        "macro_f1": metrics["macro_f1"],
                        "weighted_f1": metrics["weighted_f1"],
                        "fit_seconds": fit_seconds,
                        "prediction_seconds": prediction_seconds,
                        "latency_ms_per_document": (
                            prediction_seconds / validation_count * 1000
                        ),
                        "throughput_documents_per_second": (
                            validation_count / prediction_seconds
                            if prediction_seconds > 0
                            else None
                        ),
                        "artifact_size_bytes": artifact_size_bytes,
                    }
                )

                print(
                    f"{run_id}: "
                    f"macro-F1={metrics['macro_f1']:.4f}, "
                    f"accuracy={metrics['accuracy']:.4f}, "
                    f"fit={fit_seconds:.2f}s"
                )

    metrics_path = project_root / output_config["metrics_csv"]
    results_path = project_root / output_config["results_json"]

    _write_metrics_csv(metrics_path, summary_rows)

    results_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": "20newsgroups",
        "development_split": str(data_config["split_manifest_path"]),
        "random_seed": random_seed,
        "training_documents": len(y_train),
        "validation_documents": len(y_validation),
        "class_labels": class_labels,
        "experiment_count": len(results),
        "primary_selection_metric": "macro_f1",
        "test_data_used": False,
        "runs": results,
    }

    results_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "runs": results,
        "summary": summary_rows,
        "metrics_csv": str(metrics_path),
        "results_json": str(results_path),
    }


def generate_validation_predictions(
    config: dict[str, Any],
    *,
    project_root: Path,
    classifier_name: str,
    feature_type: str,
    representation: str,
    output_path: Path,
) -> dict[str, Any]:
    """Generate predictions for one classical model on the fixed validation split.

    Only official training-partition records are loaded. The supplied split
    manifest determines the training and validation IDs. Official test records
    are never passed to the vectorizer, classifier or evaluator.
    """
    data_config = config["data"]
    preprocessing_config = config["preprocessing"]
    feature_config = config["features"]
    classifier_config = config["classifiers"]

    raw_path = project_root / data_config["raw_dataset_path"]
    split_path = project_root / data_config["split_manifest_path"]

    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    if classifier_name not in classifier_config:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    classifier_settings = classifier_config[classifier_name]

    if not classifier_settings.get("enabled", False):
        raise ValueError(f"Classifier is not enabled: {classifier_name}")

    if feature_type not in feature_config["types"]:
        raise ValueError(f"Feature type is not configured: {feature_type}")

    if representation not in preprocessing_config["representations"]:
        raise ValueError(f"Representation is not configured: {representation}")

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
    validation_labels = {record["label"] for record in validation_records}

    if validation_labels != set(class_labels):
        raise ValueError("Training and validation must contain the same class labels.")

    lowercase = bool(preprocessing_config.get("lowercase", False))

    X_train = [
        preprocess_text(
            record["text"],
            representation=representation,
            lowercase=lowercase,
        )
        for record in training_records
    ]

    X_validation = [
        preprocess_text(
            record["text"],
            representation=representation,
            lowercase=lowercase,
        )
        for record in validation_records
    ]

    y_train = [record["label"] for record in training_records]

    vectorizer = build_vectorizer(feature_type, feature_config)
    classifier = build_classifier(
        classifier_name,
        classifier_settings,
        random_seed=int(config["project"]["random_seed"]),
    )

    pipeline = Pipeline(
        steps=[
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_validation)

    run_id = f"{classifier_name}__{feature_type}__{representation}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for record_id, record, prediction in zip(
            validation_ids,
            validation_records,
            predictions,
            strict=True,
        ):
            result = {
                "record_id": record_id,
                "actual_label": record["label"],
                "predicted_label": str(prediction),
            }

            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    return {
        "run_id": run_id,
        "classifier": classifier_name,
        "feature_type": feature_type,
        "representation": representation,
        "training_documents": len(training_records),
        "validation_documents": len(validation_records),
        "class_labels": class_labels,
        "test_data_used": False,
        "predictions_path": str(output_path),
    }
