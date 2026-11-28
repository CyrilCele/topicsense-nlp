# TopicSense-NLP — Architecture

**Project:** TopicSense-NLP  
**Version:** 1.0.0  
**Status:** Final research implementation

## 1. Overview

TopicSense-NLP is a modular Python research project for supervised multi-class text classification using the 20 Newsgroups dataset.

The implementation provides reusable functionality for dataset ingestion, validation, preprocessing, feature extraction, model training, evaluation, experiment comparison and error analysis.

The project is organised around a Python package, configuration files, research notebooks and supporting scripts.

## 2. Architectural Principles

The implementation follows these principles:

- **Modularity:** Responsibilities are separated into focused Python modules.
- **Reproducibility:** Configurations, data partitions and dependency versions are maintained explicitly.
- **Separation of concerns:** Data processing, preprocessing, modelling and evaluation are distinct.
- **Reusability:** Shared functionality is implemented in the Python package.
- **Testability:** Core functionality is supported by automated tests.
- **Research integrity:** Training, validation and official test data have distinct roles.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.12 |
| Environment management | uv |
| Numerical computing | NumPy |
| Data manipulation | pandas |
| Classical machine learning | scikit-learn |
| Neural networks | PyTorch |
| Transformer models | Hugging Face Transformers |
| Sentence embeddings | Sentence Transformers |
| Configuration | YAML / PyYAML |
| Research environment | JupyterLab |
| Visualisation | Matplotlib and Seaborn |
| Testing | pytest |
| Linting and formatting | Ruff |
| Static type checking | mypy |

The authoritative dependency definitions are `pyproject.toml` and `uv.lock`.

## 4. Repository Organisation

```text
topicsense-nlp/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       └── 20newsgroups/
│           └── splits/
├── docs/
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
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

## 5. Package Responsibilities

### Data

The data functionality handles dataset acquisition, validation, document auditing, duplicate analysis and data-quality investigations.

### Preprocessing

The preprocessing functionality prepares text and supports reproducible dataset partitioning.

### Features

The feature functionality provides numerical text representations, including Bag-of-Words and TF-IDF.

### Models

Model implementations are organised by family:

- `classical`: classical machine-learning estimators.
- `neural`: TextCNN and BiLSTM architectures.
- `transformers`: transformer-related model functionality.

### Training

The training functionality supports the training workflows for the different model families.

### Evaluation

The evaluation functionality provides classification metrics, comparative evaluation and error analysis.

## 6. Research Workflow

The project follows a sequential research workflow:

1. Dataset acquisition and exploration.
2. Data validation and quality assessment.
3. Preprocessing and partition preparation.
4. Classical machine-learning experiments.
5. Neural-network experiments.
6. Transformer experiments.
7. Model comparison.
8. Classification error analysis.

The seven numbered notebooks document this workflow.

## 7. Data Organisation

The project separates data according to its processing stage:

- `data/raw/`: acquired source data.
- `data/interim/`: intermediate processing outputs.
- `data/processed/`: prepared datasets and split information.

The processed split manifests are maintained under:

`data/processed/20newsgroups/splits/`

These include:

- `official_training_split.json`
- `decontaminated_training_split.json`
- `preprocessing_split_manifest.json`

The official test partition remains separate from model development and selection.

## 8. Experiment Outputs

Research outputs are organised under `results/`, including generated metrics, figures, tables and logs.

Outputs must be interpreted in relation to their dataset partition and experiment configuration.

## 9. Testing and Code Quality

The project uses:

- pytest for automated testing.
- Ruff for linting and formatting.
- mypy for static type checking.

These tools support the maintainability and reliability of the reusable Python package and supporting scripts.

## 10. Reproducibility

Reproducibility is supported through:

- Version-controlled dependency specifications.
- YAML experiment configurations.
- Documented dataset partitions.
- Reusable Python modules.
- Sequential research notebooks.
- Consistent evaluation procedures.
- Automated tests.

The `uv.lock` file records resolved dependency versions.

## 11. Conclusion

TopicSense-NLP provides a modular and reproducible research implementation for supervised multi-class text classification.

Its architecture reflects the completed research scope: dataset engineering, classical machine learning, neural networks, transformer experiments, model comparison and error analysis.