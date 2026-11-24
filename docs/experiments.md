# TopicSense-NLP — Experimental Protocol

**Document status:** Approved experimental baseline  
**Version:** 1.0.0

## 1. Purpose

This document defines the experimental protocol for TopicSense-NLP.

It establishes a consistent framework for model comparison, preprocessing evaluation, learning-curve analysis, cross-domain testing and computational performance measurement.

The protocol is designed to support reproducible and scientifically defensible experiments.

## 2. Experiment Design Principles

All experiments must follow these principles:

1. Use documented datasets and fixed split identifiers.
2. Prevent training and test data leakage.
3. Record all model configurations.
4. Use validation data for model selection.
5. Reserve test data for final evaluation.
6. Record computational resource usage.
7. Maintain reproducible experiment identifiers.
8. Avoid unsupported conclusions from isolated results.

## 3. Experiment Identification

Experiments will use a consistent naming convention.

| Identifier | Experiment |
|---|---|
| EXP-01 | Multinomial Naive Bayes baseline |
| EXP-02 | Logistic Regression baseline |
| EXP-03 | Linear SVM baseline |
| EXP-04 | TextCNN |
| EXP-05 | Bidirectional LSTM |
| EXP-06 | DistilBERT fine-tuning |
| EXP-07 | Frozen transformer embeddings |
| EXP-08 | Bag of Words versus TF-IDF |
| EXP-09 | Preprocessing comparison |
| EXP-10 | Training dataset size analysis |
| EXP-11 | Hyperparameter optimisation |
| EXP-12 | Cross-domain evaluation |
| EXP-13 | Confidence and calibration analysis |
| EXP-14 | Classification error analysis |
| EXP-15 | Computational efficiency comparison |

Experiments may be omitted or narrowed where resource constraints or methodological limitations justify the decision. Any omission must be documented.

## 4. Classical Baseline Experiments

### EXP-01: Multinomial Naive Bayes

Objective: Establish a probabilistic classical classification baseline.

Initial feature representation: TF-IDF or count-based features, selected and documented during implementation.

### EXP-02: Logistic Regression

Objective: Evaluate a linear discriminative classifier.

The experiment will investigate regularisation and relevant solver configurations.

### EXP-03: Linear Support Vector Machine

Objective: Evaluate a linear margin-based classifier.

The experiment will investigate regularisation strength and decision-score behaviour.

The three classical models must use comparable datasets and evaluation splits.

## 5. Neural Network Experiments

### EXP-04: TextCNN

Objective: Investigate convolutional neural networks for text classification.

The experiment will document:

- Tokenisation.
- Vocabulary size.
- Embedding dimension.
- Convolutional filter configuration.
- Optimiser.
- Learning rate.
- Batch size.
- Epoch count.
- Early stopping strategy.

### EXP-05: Bidirectional LSTM

Objective: Investigate sequential neural representations.

This experiment is conditional on available CPU time and memory.

It must use a comparable evaluation protocol to the other models.

## 6. Transformer Experiments

### EXP-06: DistilBERT

Objective: Evaluate transformer fine-tuning for multi-class topic classification.

Initial CPU smoke-test configuration:

- Maximum sequence length: 128.
- Batch size: 4.
- Gradient accumulation: 2.
- Epochs: 2.
- Learning rate: 2e-5.
- Weight decay: 0.01.
- Warmup ratio: 0.1.

These values are starting points for a smoke test, not guaranteed optimal hyperparameters.

The experiment must record runtime and memory consumption.

### EXP-07: Frozen Transformer Embeddings

Objective: Investigate the use of pretrained sentence embeddings with a lightweight downstream classifier.

This experiment is optional and depends on available resources.

## 7. Feature Representation Experiments

### EXP-08: Bag of Words versus TF-IDF

Compare the effects of count-based and TF-IDF representations.

The comparison must control for the classifier and data split.

Where practical, repeat the comparison across the three classical baselines.

## 8. Preprocessing Experiments

### EXP-09: Preprocessing Comparison

Investigate the effects of selected preprocessing decisions.

Potential comparisons:

- Raw versus normalised text.
- Lowercase versus case-preserving text.
- Stopword removal versus retention.
- Stemming versus lemmatisation.
- Unigrams versus unigram/bigram features.

Only meaningful, controlled comparisons should be included.

## 9. Training Dataset Size Experiments

### EXP-10: Learning Curves

Investigate the relationship between training data size and classification performance.

Proposed training fractions:

| Fraction | Purpose |
|---|---|
| 10% | Small-data behaviour |
| 25% | Early learning behaviour |
| 50% | Intermediate learning behaviour |
| 75% | Near-full training behaviour |
| 100% | Full training performance |

The validation and test sets must remain fixed.

Training subsets must be sampled reproducibly and preserve class representation where possible.

## 10. Hyperparameter Optimisation

### EXP-11: Optuna Optimisation

Use Optuna selectively to optimise model configurations.

The optimisation objective should be defined before the search.

The validation set must be used for optimisation, never the final test set.

Search spaces, trial counts, random seeds and stopping criteria must be recorded.

## 11. Cross-Domain Evaluation

### EXP-12: Domain Transfer

Evaluate models across compatible datasets from different textual domains.

Record:

- Source training dataset.
- Target evaluation dataset.
- Label mapping.
- Source and target class distributions.
- In-domain metrics.
- Cross-domain metrics.
- Performance differences.

Do not compare incompatible label taxonomies as if they represented identical tasks.

## 12. Confidence and Calibration

### EXP-13: Confidence Analysis

Investigate prediction confidence and calibration where model outputs support meaningful probability estimates.

Potential analyses:

- Reliability diagrams.
- Calibration error.
- Temperature scaling.
- Confidence distributions.
- Selective prediction.

Do not interpret raw SVM decision scores as calibrated probabilities.

## 13. Error Analysis

### EXP-14: Error Analysis

Investigate:

- Frequently confused classes.
- False positives.
- False negatives.
- Ambiguous documents.
- Short and long documents.
- Domain-specific errors.
- Class imbalance effects.

Examples selected for reporting must be handled responsibly and must not expose unnecessary personal information.

## 14. Computational Efficiency

### EXP-15: Efficiency Comparison

Measure:

- Training duration.
- Inference latency.
- Throughput.
- Peak memory where measurable.
- Model artifact size.

Comparisons must document the measurement environment, batch size and timing method.

Do not compare runtimes collected under materially different conditions without explaining the difference.

## 15. Experiment Tracking

MLflow will be used to record:

- Experiment name.
- Run identifier.
- Dataset and split.
- Parameters.
- Metrics.
- Artifacts.
- Runtime information.
- Model references.

Experiment outputs will be organised under:

```text
results/
├── metrics/
├── figures/
├── tables/
└── logs/
```

Model checkpoints and registry artifacts will be organised under:

```text
models/
├── checkpoints/
└── registry/
```

## 16. Statistical Interpretation

Results must be interpreted in relation to dataset size, class distribution, domain characteristics and model complexity.

Where computationally practical, repeated runs or confidence intervals should be used to assess result stability.

Small performance differences must not automatically be presented as meaningful improvements.

## 17. Reporting Requirements

Every reported experiment must include:

- Research objective.
- Dataset and split.
- Model configuration.
- Experimental conditions.
- Evaluation metrics.
- Computational measurements where applicable.
- Observed results.
- Interpretation.
- Limitations.

No experiment may be reported as completed before it has actually been executed and validated.