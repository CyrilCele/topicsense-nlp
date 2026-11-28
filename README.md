# TopicSense-NLP

**Reproducible research on supervised multi-class text classification using the 20 Newsgroups dataset.**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Overview

TopicSense-NLP is a reproducible natural language processing research project investigating supervised text classification. It compares classical machine-learning models, neural networks, and transformer-based approaches on the 20 Newsgroups dataset.

The project examines how model architecture, text representation, and preprocessing choices affect multi-class classification performance. Experiments are organised into Jupyter notebooks, with reusable functionality implemented in a Python package.

## 2. Research Objectives

- Explore and validate the 20 Newsgroups dataset.
- Establish classical machine-learning baselines.
- Evaluate neural-network approaches to text classification.
- Investigate transformer-based representations and fine-tuning.
- Compare model performance using consistent evaluation metrics.
- Examine classification errors and identify patterns in model performance.
- Maintain reproducible data processing and experiment organisation.

## 3. Dataset

The project uses the **20 Newsgroups dataset**, a collection of newsgroup documents organised into 20 topic categories.

| Dataset characteristic | Value |
|:---|---:|
| Number of documents | 18,846 |
| Number of classes | 20 |
| Original training documents | 11,314 |
| Original test documents | 7,532 |
| Development training documents | 9,051 |
| Validation documents | 2,263 |

The original training and test partitions are preserved. Model development and comparison use the training and validation partitions. The official test partition is kept separate from model selection.

Dataset preprocessing and split information are maintained under `data/`.

## 4. Models and Experiments

The experiments investigate three model families.

### 4.1 Classical Machine Learning

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine

These models are evaluated using Bag-of-Words and TF-IDF representations, with experiments considering full-message and message-body text.

### 4.2 Neural Networks

- TextCNN
- Bidirectional LSTM (BiLSTM)

### 4.3 Transformer-Based Approaches

- Fine-tuned DistilBERT
- Frozen Sentence Transformer embeddings with Logistic Regression

The experiments compare 16 configurations across the three model families.

## 5. Evaluation

Model performance is assessed using standard multi-class classification metrics:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion matrices
- Class-level classification reports

The validation set is used for model comparison and error analysis. Validation results must not be interpreted as performance on the untouched official test set.

The strongest validation configuration recorded in the experiments is **Linear SVM with TF-IDF features using full-message text**, achieving:

| Metric | Validation result |
|:---|---:|
| Accuracy | 91.47% |
| Macro F1-score | 0.9138 |

These figures are validation results, not official test-set results.

## 6. Repository Structure

```text
topicsense-nlp/
├── configs/
│   ├── data.yaml
│   ├── classical_ml.yaml
│   ├── neural_networks.yaml
│   ├── transformers.yaml
│   └── experiments.yaml
├── data/
│   ├── raw/
│   │   └── 20newsgroups/
│   ├── interim/
│   │   └── 20newsgroups/
│   └── processed/
│       └── 20newsgroups/
│           └── splits/
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_classical_baselines.ipynb
│   ├── 04_neural_networks.ipynb
│   ├── 05_transformer_experiments.ipynb
│   ├── 06_model_comparison.ipynb
│   └── 07_error_analysis.ipynb
├── scripts/
├── src/
│   └── topicsense/
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
│       ├── experiments/
│       ├── explainability/
│       ├── inference/
│       ├── registry/
│       └── utils/
├── tests/
├── results/
├── docs/
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

## 7. Technology Stack

| Technology | Purpose |
|:---|:---|
| Python 3.12 | Primary programming language |
| uv | Dependency and environment management |
| NumPy | Numerical computing |
| pandas | Data manipulation |
| scikit-learn | Classical machine learning and evaluation |
| PyTorch | Neural-network modelling |
| Hugging Face Transformers | Transformer architectures and fine-tuning |
| Sentence Transformers | Sentence embeddings |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| JupyterLab | Interactive research notebooks |
| pytest | Automated testing |
| Ruff | Linting and code formatting |
| mypy | Static type checking |

## 8. Getting Started

### 8.1 Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

### 8.2 Installation

Clone the repository:

```bash
git clone https://github.com/CyrilCele/topicsense-nlp.git
cd topicsense-nlp
```

Install the project and its dependency groups:

```bash
uv sync --all-groups
```

### 8.3 Run the Notebooks

Start JupyterLab:

```bash
uv run jupyter lab
```

Open the notebooks in numerical order, beginning with dataset exploration.

### 8.4 Run Tests

```bash
uv run pytest
```

### 8.5 Run Code-Quality Checks

```bash
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run mypy src scripts
```

## 9. Reproducibility

The project uses configuration files, version-controlled dependency specifications, organised dataset partitions, reusable Python modules, and sequential research notebooks to support reproducibility.

The `uv.lock` file records the resolved dependency versions. Experiment configurations are maintained in `configs/`, while reusable implementation code is maintained in `src/topicsense/`.

## 10. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 11. Author

**Nkululeko Cyril Cele**

- GitHub: [@CyrilCele](https://github.com/CyrilCele)
- Project: [TopicSense-NLP](https://github.com/CyrilCele/topicsense-nlp)

TopicSense-NLP is an academic and professional portfolio project demonstrating supervised natural language processing, classical machine learning, deep learning, transformer architectures, systematic model evaluation, error analysis, and reproducible experimentation.

---

**TopicSense-NLP — Research rigor. Reproducible engineering. Measurable intelligence.**