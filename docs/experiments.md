# TopicSense-NLP — Experiment Documentation

## 1. Overview

This document describes the experimental organisation of TopicSense-NLP.

The project investigates supervised multi-class text classification using the 20 Newsgroups dataset.

The experiments compare classical machine-learning models, neural networks and transformer-based approaches.

A total of 16 configurations are included in the experimental scope.

## 2. Experimental Dataset

The experiments use the 20 Newsgroups dataset.

| Partition | Documents |
|---|---:|
| Development training | 9,051 |
| Validation | 2,263 |
| Official test | 7,532 |

The development training partition is used for fitting models.

The validation partition supports model comparison and error analysis.

The official test partition is kept separate from model selection.

## 3. Notebook Sequence

The research is documented through seven sequential notebooks.

| Notebook | Research stage |
|---|---|
| `01_dataset_exploration.ipynb` | Dataset exploration |
| `02_data_preprocessing.ipynb` | Data preparation and preprocessing |
| `03_classical_baselines.ipynb` | Classical machine-learning experiments |
| `04_neural_networks.ipynb` | Neural-network experiments |
| `05_transformer_experiments.ipynb` | Transformer experiments |
| `06_model_comparison.ipynb` | Comparative model evaluation |
| `07_error_analysis.ipynb` | Classification error analysis |

The sequence follows the research workflow from dataset understanding through model evaluation.

## 4. Classical Machine-Learning Experiments

The classical experiments evaluate three algorithms.

| Algorithm | Description |
|---|---|
| Multinomial Naive Bayes | Probabilistic text classification |
| Logistic Regression | Linear classification |
| Linear SVM | Maximum-margin linear classification |

Two feature representations are investigated:

- Bag-of-Words.
- TF-IDF.

Two text scopes are considered:

- Full-message text.
- Message-body text.

This produces 12 classical experimental configurations.

## 5. Neural-Network Experiments

The neural experiments investigate:

| Architecture | Description |
|---|---|
| TextCNN | Convolutional text classification |
| BiLSTM | Bidirectional recurrent text classification |

Two neural-network configurations are included.

## 6. Transformer Experiments

Two transformer-based approaches are investigated.

### 6.1 DistilBERT Fine-Tuning

DistilBERT is adapted to the 20 Newsgroups classification task through supervised fine-tuning.

### 6.2 Frozen Sentence Transformer Embeddings

The `sentence-transformers/all-MiniLM-L6-v2` model produces sentence embeddings.

These embeddings are used with Logistic Regression for classification.

The embedding model remains frozen.

Two transformer configurations are included.

## 7. Experiment Summary

| Model family | Number of configurations |
|---|---:|
| Classical machine learning | 12 |
| Neural networks | 2 |
| Transformer-based approaches | 2 |
| **Total** | **16** |

## 8. Evaluation Metrics

The experiments use the following classification metrics:

- Accuracy.
- Precision.
- Recall.
- Macro F1-score.
- Confusion matrices.
- Class-level classification reports.

Macro F1-score is particularly useful for comparing performance across the 20 categories.

## 9. Model Comparison

The model comparison stage consolidates the experimental results.

The recorded strongest validation configuration is:

**Linear SVM + TF-IDF + full-message text**

| Metric | Validation result |
|---|---:|
| Accuracy | 91.47% |
| Macro F1-score | 0.9138 |

These are validation metrics and must not be presented as official test-set results.

## 10. Error Analysis

The error-analysis stage examines misclassified validation documents.

The analysis considers:

- Actual labels.
- Predicted labels.
- Misclassification patterns.
- Class-level performance.
- Potentially ambiguous examples.

The purpose is to complement aggregate evaluation metrics with a more detailed understanding of classification behaviour.

## 11. Configuration Management

Experiment settings are maintained in the `configs/` directory.

The configuration files cover:

- Data settings.
- Classical machine-learning settings.
- Neural-network settings.
- Transformer settings.
- Experiment settings.

The configuration files provide a central location for experiment parameters and support consistent execution.

## 12. Reproducibility

Experimental reproducibility is supported through:

- Version-controlled code.
- `uv.lock`.
- Configuration files.
- Documented dataset partitions.
- Sequential notebooks.
- Reusable model implementations.
- Consistent evaluation metrics.

## 13. Interpretation and Limitations

The results represent experiments on one historical text-classification dataset.

Performance should not be generalised automatically to other datasets or domains.

Validation metrics are used for model comparison. They are not substitutes for independent official test-set evaluation.

## 14. Conclusion

The experimental framework provides a structured comparison of 16 configurations across classical machine learning, neural networks and transformer-based approaches.

The resulting analysis supports an empirical understanding of text representation, model architecture and classification performance.