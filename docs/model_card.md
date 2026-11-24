# TopicSense-NLP — Model Card

**Document status:** Initial template — Model development pending  
**Version:** 1.0.0

## 1. Model Overview

TopicSense-NLP will contain multiple supervised text classification models.

The purpose of this document is to describe model design, intended use, evaluation, limitations and operational considerations.

A separate model entry should be maintained for every materially different trained model or model version.

## 2. Model Inventory

| Model | Family | Status |
|---|---|---|
| Multinomial Naive Bayes | Classical ML | Planned |
| Logistic Regression | Classical ML | Planned |
| Linear SVM | Classical ML | Planned |
| TextCNN | Neural network | Planned |
| Bidirectional LSTM | Neural network | Conditional |
| DistilBERT | Transformer | Planned |
| Frozen transformer embeddings | Transformer representation | Optional |

No model is considered trained or validated merely because it appears in this inventory.

## 3. Intended Use

The intended primary use is supervised multi-class classification of textual documents into predefined topic categories.

Potential application capabilities include:

- Single-document topic prediction.
- Batch document classification.
- Prediction confidence or scores where supported.
- Model comparison.
- Analytical reporting.

The system is intended for research, experimentation and demonstration.

It is not intended to make high-stakes decisions without additional validation and safeguards.

## 4. Training Data

For each model, document:

- Dataset name and version.
- Dataset source.
- Licence.
- Training split identifier.
- Class distribution.
- Preprocessing configuration.
- Feature representation.
- Random seed.

Dataset details must reference the corresponding dataset documentation.

## 5. Model Architecture

Record the relevant architecture and configuration.

For classical models:

- Algorithm.
- Feature extraction method.
- Vocabulary configuration.
- Regularisation or smoothing parameters.
- Class weighting, if used.

For neural models:

- Embedding configuration.
- Architecture.
- Optimiser.
- Learning rate.
- Batch size.
- Epoch count.
- Regularisation.
- Early stopping configuration.

For transformer models:

- Model identifier.
- Pretrained checkpoint.
- Tokeniser.
- Maximum sequence length.
- Fine-tuning configuration.
- Training hardware.
- Runtime and memory measurements.

## 6. Evaluation

Each model must be evaluated using held-out data.

Required metrics:

- Accuracy.
- Macro precision.
- Macro recall.
- Macro F1-score.
- Weighted F1-score.
- Per-class precision, recall and F1-score.
- Confusion matrix.

Additional metrics may include:

- Calibration.
- Inference latency.
- Throughput.
- Training duration.
- Memory usage.
- Artifact size.

Actual metric values must only be entered after the relevant evaluation has been completed.

## 7. Performance Record

| Metric | Result |
|---|---|
| Accuracy | Pending evaluation |
| Macro precision | Pending evaluation |
| Macro recall | Pending evaluation |
| Macro F1-score | Pending evaluation |
| Weighted F1-score | Pending evaluation |
| Training duration | Pending measurement |
| Inference latency | Pending measurement |
| Peak memory | Pending measurement |
| Artifact size | Pending measurement |

## 8. Limitations

Potential model limitations include:

- Dependence on training-data quality.
- Class imbalance.
- Domain shift.
- Vocabulary mismatch.
- Ambiguous documents.
- Out-of-domain inputs.
- Computational limitations.
- Differences in confidence-score interpretation across model families.

These limitations must be evaluated and refined after training.

## 9. Bias and Fairness Considerations

Training data may contain historical, demographic, cultural or domain-specific biases.

Evaluation should investigate whether model errors are concentrated in particular classes or text categories.

The system must not be assumed to be unbiased merely because it achieves high aggregate accuracy.

## 10. Confidence and Prediction Interpretation

Prediction scores must be interpreted according to the underlying model.

Probabilities from probabilistic classifiers, calibrated probabilities and raw decision scores are not interchangeable.

Where confidence is exposed to users, its meaning and limitations must be documented.

## 11. Model Versioning and Tracking

Model records should include:

- Model name.
- Model version.
- Training run identifier.
- Dataset and split.
- Configuration.
- Evaluation metrics.
- Artifact location.
- Creation timestamp.
- Software environment.

MLflow will provide experiment tracking and model lifecycle integration.

## 12. Computational Requirements

The project targets CPU-only development.

Classical models are expected to provide the initial reference points.

Neural and transformer models must be evaluated with explicit resource limits.

Actual runtime and memory requirements must be measured rather than assumed.

## 13. Ethical and Operational Considerations

Predictions are statistical estimates and may be incorrect.

The application should provide appropriate warnings where classification confidence is weak or inputs fall outside the intended domain.

Raw user text must not be retained unnecessarily.

## 14. Model Status

**Current status:** Model implementations and evaluations pending.

This document defines the reporting standard for future models. It does not claim that any model has already been trained, evaluated or approved for deployment.