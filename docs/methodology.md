# TopicSense-NLP — Research Methodology

## 1. Introduction

This document describes the methodology adopted in TopicSense-NLP, a supervised multi-class text classification research project using the 20 Newsgroups dataset.

The research compares classical machine-learning algorithms, neural-network architectures and transformer-based approaches under a consistent experimental framework.

The methodology covers dataset preparation, text representation, model development, evaluation and error analysis.

## 2. Research Design

The project follows an experimental comparative research design.

Three model families are investigated:

1. Classical machine learning.
2. Neural networks.
3. Transformer-based approaches.

The objective is to examine the impact of model architecture and text representation on multi-class classification performance.

The experiments use the same underlying classification task and documented data partitions.

## 3. Dataset

The 20 Newsgroups dataset contains 18,846 documents belonging to 20 topic categories.

The original partitions are:

| Partition | Documents |
|---|---:|
| Original training | 11,314 |
| Official test | 7,532 |
| Total | 18,846 |

The original training partition is divided into:

| Development partition | Documents |
|---|---:|
| Training | 9,051 |
| Validation | 2,263 |

The official test partition is retained separately from model development and selection.

## 4. Data Preparation

Data preparation involves:

- Dataset acquisition.
- Dataset validation.
- Document inspection.
- Duplicate analysis.
- Contamination and leakage investigation.
- Text preprocessing.
- Dataset partition management.

The project separates raw, intermediate and processed data.

Split manifests are maintained under:

`data/processed/20newsgroups/splits/`

The preprocessing workflow also considers two text scopes:

- Full-message text.
- Message-body text.

This allows the experiments to investigate the influence of document headers and message content on classification performance.

## 5. Classical Machine Learning

The classical machine-learning experiments investigate three algorithms:

### 5.1 Multinomial Naive Bayes

Multinomial Naive Bayes provides a probabilistic baseline for text classification using non-negative feature representations.

### 5.2 Logistic Regression

Logistic Regression provides a linear classification approach that estimates class probabilities and supports multi-class prediction.

### 5.3 Linear Support Vector Machine

Linear SVM learns separating hyperplanes in a high-dimensional feature space and provides a strong baseline for sparse text representations.

### 5.4 Feature Representations

The classical models are evaluated using:

- Bag-of-Words.
- Term Frequency-Inverse Document Frequency (TF-IDF).

Both full-message and message-body text are considered.

The classical experiments contain 12 configurations.

## 6. Neural-Network Experiments

Two neural-network architectures are investigated.

### 6.1 TextCNN

TextCNN uses convolutional operations over token representations to identify local textual patterns relevant to classification.

### 6.2 Bidirectional LSTM

BiLSTM processes sequences in both forward and backward directions, allowing contextual information from both directions to contribute to document representations.

The neural-network experiments contain two configurations.

## 7. Transformer Experiments

The transformer experiments investigate two approaches.

### 7.1 Fine-Tuned DistilBERT

DistilBERT is fine-tuned for supervised classification using the project dataset.

This approach investigates contextual representations learned through transformer pretraining and task-specific adaptation.

### 7.2 Frozen Sentence Transformer Embeddings

The `sentence-transformers/all-MiniLM-L6-v2` model generates sentence embeddings.

The embeddings are used as input features for Logistic Regression.

The embedding model remains frozen during this classification workflow.

The transformer experiments contain two configurations.

## 8. Experimental Scope

The complete experimental comparison contains 16 configurations:

| Model family | Configurations |
|---|---:|
| Classical machine learning | 12 |
| Neural networks | 2 |
| Transformer-based approaches | 2 |
| **Total** | **16** |

The configurations are maintained in the project configuration files and associated research workflows.

## 9. Evaluation Methodology

Classification performance is evaluated using:

- Accuracy.
- Precision.
- Recall.
- Macro F1-score.
- Confusion matrices.
- Class-level classification reports.

Macro-averaged metrics provide insight into performance across all classes without weighting classes according to their sample counts.

The validation partition supports model comparison and selection.

The official test partition remains separate from these activities.

## 10. Model Comparison

The model comparison process consolidates experimental results to examine differences between model families and configurations.

The comparison considers:

- Model architecture.
- Text representation.
- Text scope.
- Classification accuracy.
- Macro F1-score.
- Class-level performance.
- Computational considerations.

The strongest recorded validation configuration is Linear SVM with TF-IDF features using full-message text.

Its recorded validation performance is:

| Metric | Result |
|---|---:|
| Accuracy | 91.47% |
| Macro F1-score | 0.9138 |

These values represent validation performance, not official test-set performance.

## 11. Error Analysis

Error analysis investigates incorrectly classified validation examples.

The analysis considers:

- Actual and predicted categories.
- Misclassification patterns.
- Class-level performance differences.
- Potentially ambiguous documents.
- The relationship between document content and classification errors.

The purpose is to understand model behaviour rather than relying exclusively on aggregate performance metrics.

## 12. Reproducibility

The research methodology supports reproducibility through:

- Explicit dataset partitions.
- Version-controlled dependency specifications.
- YAML experiment configurations.
- Reusable Python implementations.
- Sequentially organised notebooks.
- Consistent evaluation procedures.
- Automated tests.

The project uses `uv.lock` to record resolved dependency versions.

## 13. Methodological Limitations

The findings are subject to several limitations:

- All experiments use a single dataset.
- The dataset represents historical newsgroup discussions.
- Model performance may vary across unrelated domains.
- Validation results do not establish official test-set performance.
- Different model families have different computational requirements.
- Results depend on preprocessing, feature representation and experimental configuration.

Consequently, conclusions should remain within the scope of the implemented experiments.

## 14. Conclusion

The methodology establishes a structured comparative framework for supervised multi-class text classification.

By investigating classical machine learning, neural networks and transformer-based approaches, TopicSense-NLP provides an empirical examination of different modelling strategies using a common dataset and evaluation framework.