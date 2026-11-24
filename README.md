# TopicSense NLP

**A supervised multi-domain text classification system combining classical machine learning and transformer-based deep learning.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Project-Research%20%26%20Development-orange.svg)](#project-status)

## 1. Project Overview

TopicSense NLP is a natural language processing project focused on the automatic classification of textual documents into predefined topic categories.

The project investigates and compares traditional machine learning algorithms and transformer-based deep learning approaches within a reproducible text classification framework.

The intended application considers multiple types of textual content, including news articles, academic publications and social media text. These domains will be investigated using suitable labelled datasets, subject to data availability, licensing conditions and the compatibility of their classification schemes.

The central research focus is to evaluate how modelling approaches differ in predictive performance, generalisation capability, computational requirements and sensitivity to training dataset size.

**Important terminology:** Although the assignment may use the term *topic modelling*, this implementation focuses on supervised topic classification: learning from labelled documents to predict predefined categories. It is therefore distinct from unsupervised topic modelling techniques such as Latent Dirichlet Allocation (LDA).

## 2. Research Objectives

The project aims to:

1. Identify, acquire and document appropriate labelled text datasets.
2. Investigate dataset structure, quality, class distributions, linguistic characteristics and potential biases.
3. Develop model-appropriate text cleaning, normalization and preprocessing pipelines.
4. Establish reproducible classical machine learning baselines.
5. Investigate transformer-based deep learning for supervised text classification.
6. Apply systematic hyperparameter optimization using validation data.
7. Evaluate predictive performance, generalisation and computational efficiency.
8. Investigate the relationship between training dataset size and model performance.
9. Examine classification errors, model limitations and potential sources of bias.
10. Document the research methodology, experimental results and conclusions.

## 3. Modelling Methodology

The project will investigate two complementary modelling approaches.

### 3.1 Classical Machine Learning

The planned classical algorithms are:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (Linear SVM)

Text representation will primarily use Term Frequency–Inverse Document Frequency (TF-IDF).

The preprocessing and feature extraction pipeline will be developed to suit traditional machine learning algorithms. Relevant experiments may investigate token normalization, stopword handling, n-gram representations and vocabulary filtering.

### 3.2 Transformer-Based Deep Learning

The planned deep learning approach is:

- DistilBERT fine-tuning using PyTorch and Hugging Face Transformers.

The transformer pipeline will use the tokenizer and input representation associated with the selected pretrained model.

Unlike the classical pipeline, transformer inputs will not automatically undergo operations such as stemming, lemmatization or indiscriminate stopword removal, since these transformations may remove useful contextual information.

### 3.3 Model Comparison

The approaches will be compared using consistent data partitions, appropriate evaluation metrics and documented experimental conditions.

Model complexity alone will not determine model selection. Conclusions will be based on measured performance, generalisation and computational cost.

## 4. Data Strategy

Datasets will be selected according to:

- Relevance to the classification objectives.
- Availability of reliable topic labels.
- Dataset size and class distribution.
- Text quality and completeness.
- Domain representation.
- Licensing and redistribution permissions.
- Compatibility of category definitions.

Potential dataset sources will be investigated during data acquisition and understanding.

Cross-domain evaluation will only be performed where category definitions can be mapped in a methodologically defensible manner. Incompatible label schemes will be evaluated separately rather than treated as equivalent.

The project will maintain a clear separation between:

- **Raw data:** Original datasets, preserved without modification.
- **Interim data:** Intermediate outputs generated during data preparation.
- **Processed data:** Validated datasets prepared for modelling.
- **External data:** Supporting resources and supplementary datasets.

## 5. Experimental Design and Evaluation

### 5.1 Data Partitioning

The experimental design will use training, validation and test partitions.

Where suitable official benchmark splits exist, these will be preserved where possible.

The validation set will support model selection and hyperparameter optimization. The test set will remain independent of model development and will be used for final performance evaluation.

Data leakage will be prevented by fitting learned preprocessing operations, including feature extraction and vocabulary construction, exclusively on the training data.

### 5.2 Evaluation Metrics

The evaluation will investigate:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Per-class precision, recall and F1-score
- Confusion matrices
- Training time
- Inference time
- Model size and computational requirements

Macro F1-score will be the primary comparative metric where class imbalance makes overall accuracy insufficient.

Additional analyses may include confidence calibration, uncertainty, statistical variability and error analysis where justified by the experimental design.

### 5.3 Training Dataset Size Experiments

Model performance will be investigated at progressively increasing training dataset sizes:

- 10%
- 25%
- 50%
- 75%
- 100%

Validation and test partitions will remain fixed across these experiments to support meaningful comparisons.

### 5.4 Hyperparameter Optimization

Optuna will be investigated for systematic hyperparameter optimization.

Optimization will use validation performance rather than test-set results.

The search spaces will be tailored to the individual algorithms and computational resources available.

## 6. Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python 3.12 |
| Environment and dependency management | uv |
| Data manipulation | NumPy, Pandas |
| Classical machine learning | Scikit-learn |
| Deep learning | PyTorch |
| NLP and pretrained models | Hugging Face Transformers |
| Hyperparameter optimization | Optuna |
| Experiment tracking | MLflow |
| Research environment | JupyterLab |
| Visualization | Matplotlib, Seaborn |
| Testing | Pytest |
| Code quality | Ruff, Mypy |
| Version control | Git, GitHub, Git LFS |

Additional dependencies will be introduced when required by the implementation.

## 7. Repository Structure

```text
topicsense-nlp/
│
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
│
├── configs/
│   ├── data.yaml
│   ├── classical_ml.yaml
│   ├── neural_networks.yaml
│   ├── transformers.yaml
│   └── experiments.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_classical_baselines.ipynb
│   ├── 04_neural_networks.ipynb
│   ├── 05_transformer_experiments.ipynb
│   ├── 06_model_comparison.ipynb
│   └── 07_error_analysis.ipynb
│
├── src/
│   └── topicsense/
│       ├── __init__.py
│       ├── config/
│       ├── data/
│       ├── preprocessing/
│       ├── features/
│       ├── models/
│       │   ├── classical/
│       │   ├── neural/
│       │   └── transformers/
│       ├── training/
│       ├── evaluation/
│       ├── explainability/
│       ├── inference/
│       ├── registry/
│       └── utils/
│
├── scripts/
│   ├── download_data.py
│   ├── validate_data.py
│   ├── train_classical.py
│   ├── train_neural.py
│   ├── train_transformer.py
│   ├── evaluate_models.py
│   └── run_experiments.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── test_prediction.py
│
├── models/
│   ├── checkpoints/
│   └── registry/
│
├── results/
│   ├── metrics/
│   ├── figures/
│   ├── tables/
│   └── logs/
│
├── reports/
│   ├── technical/
│   └── academic/
│
└── docs/
    ├── architecture.md
    ├── methodology.md
    ├── experiments.md
    ├── dataset_card.md
    └── model_card.md
```

## 8. Reproducibility and Data Governance

The project will follow reproducible research and software engineering practices.

These include:

- Documented dataset sources, versions and acquisition procedures.
- Controlled random seeds and reproducible experiment configurations.
- Consistent training, validation and test procedures.
- Version-controlled source code and project documentation.
- Explicit dependency management and lockfiles.
- Automated code quality checks and testing.
- Documented model configurations and experimental results.

Dataset licences and redistribution conditions will be reviewed before datasets are committed to the public repository.

Sensitive information, credentials, API keys and private configuration files must never be committed.

Large datasets and model artifacts will be managed using Git LFS where appropriate. Data and artifacts will be tracked selectively according to licensing, storage and repository constraints.

## 9. Project Status

**Current phase:** Initial project setup.

| Project component | Status |
|---|---|
| WSL2 development environment | Completed |
| Git and GitHub configuration | Completed |
| Initial repository structure | Completed |
| Python environment | Completed |
| Dataset acquisition | Pending |
| Data understanding | Pending |
| Data preparation | Pending |
| Classical machine learning | Pending |
| Transformer-based deep learning | Pending |
| Hyperparameter optimization | Pending |
| Model evaluation | Pending |
| Final research report | Pending |

The project status will be updated as implementation progresses.

## 10. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

The MIT License applies to the original code and documentation in this repository. It does not automatically grant rights to third-party datasets, pretrained models, libraries or other external resources.

All third-party resources remain subject to their respective licences, terms of use and redistribution conditions. Dataset sources and applicable licensing information will be documented as part of the data acquisition process.


## 11. Author

**Nkululeko Cyril Cele**

- GitHub: [@CyrilCele](https://github.com/CyrilCele)
- Project: TopicSense NLP

This project was developed as an academic and professional portfolio project exploring supervised text classification, classical machine learning and transformer-based deep learning.

## Recommended additional dependencies
Recommended additional dependencies
Dependency	Purpose	Priority
tqdm	Training and data-processing progress bars	Essential
psutil	RAM and CPU monitoring	Recommended
captum	Neural-network explainability	Recommended
tensorboard	Training curves and experiment monitoring	Recommended
sentence-transformers	Compact pretrained embedding experiments	Optional
streamlit	Interactive TopicSense application	Later
plotly	Interactive evaluation dashboard	Later