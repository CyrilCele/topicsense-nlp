# TopicSense-NLP — System Architecture

**Document status:** Approved — Baseline  
**Version:** 1.0.0  
**Project:** TopicSense-NLP  
**Last updated:** 24 November 2026

## 1. Project Overview

TopicSense-NLP is a modular Natural Language Processing (NLP) system for supervised, multi-class topic classification across multiple textual domains.

The project combines classical machine learning, neural networks and transformer-based approaches within a reproducible experimental framework.

The system has two primary objectives:

1. Conduct rigorous research comparing supervised text classification approaches across different datasets and textual domains.
2. Develop a reusable NLP application capable of processing text, generating topic predictions, evaluating models and presenting analytical results.

Although the assignment refers to topic modeling, the implementation focuses on supervised topic classification using labelled documents. Unsupervised topic modeling is not a replacement for the required classification task.

## 2. Architectural Principles

The architecture follows these principles:

- **Modularity:** Each component has a clearly defined responsibility.
- **Separation of concerns:** Data processing, model training, evaluation and inference remain independent.
- **Reusability:** Core NLP functionality must be accessible from scripts and notebooks without duplicated implementations.
- **Reproducibility:** Experiments must be configurable, traceable and repeatable.
- **Testability:** Components must be designed for independent testing.
- **Scalability:** The architecture should support additional datasets, models and domains.
- **Resource awareness:** Development and experimentation must accommodate CPU-only hardware.
- **Maintainability:** Code and documentation must remain understandable and consistently organised.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.12 |
| Dependency management | uv |
| Numerical computing | NumPy |
| Data manipulation | Pandas |
| Classical machine learning | Scikit-learn |
| Scientific computing | SciPy |
| Neural networks | PyTorch |
| Transformer models | Hugging Face Transformers |
| Dataset integration | Hugging Face Datasets |
| Hyperparameter optimisation | Optuna |
| Configuration | YAML / PyYAML |
| Experiment tracking | MLflow |
| Visualisation | Matplotlib, Seaborn |
| Interactive visualisation | Plotly |
| Backend API | FastAPI |
| Frontend dashboard | Streamlit |
| Relational database | PostgreSQL |
| Database ORM | SQLAlchemy |
| Database migrations | Alembic |
| PostgreSQL driver | psycopg 3 |
| Testing | pytest |
| Code quality | Ruff |
| Static typing | mypy |
| Containerisation | Docker |
| Container orchestration | Docker Compose |
| CI/CD | GitHub Actions |
| Container registry | GitHub Container Registry |

All dependency versions are controlled through `pyproject.toml` and `uv.lock`.

## 4. Repository Architecture

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

## 5. Core Package Responsibilities

### config/

Responsible for loading, validating and managing project configurations.

### data/

Responsible for:

- Dataset acquisition.
- Dataset validation.
- Dataset standardisation.
- Dataset splitting.
- Duplicate detection.
- Dataset provenance and metadata.

### preprocessing/

Responsible for:

- Text normalisation.
- Tokenisation.
- Stopword handling.
- Stemming and lemmatisation.
- Text cleaning.
- Preprocessing pipelines.

### features/

Responsible for numerical representations of text, including:

- Bag of Words.
- TF-IDF.
- Unigrams and bigrams.
- Vocabulary management.
- Feature transformation.

### models/

Contains model implementations organised by family.

**classical/**

- Multinomial Naive Bayes.
- Logistic Regression.
- Linear Support Vector Machine.

**neural/**

- TextCNN.
- Bidirectional LSTM, subject to feasibility.

**transformers/**

- DistilBERT fine-tuning.
- Optional frozen transformer embeddings.

### training/

Responsible for training orchestration, reproducibility, hyperparameters and model checkpoint management.

### evaluation/

Responsible for performance metrics, confusion matrices, classification reports, learning curves and error analysis.

### explainability/

Responsible for model interpretation and feature attribution where methodologically justified.

### inference/

Responsible for loading trained models and providing consistent prediction interfaces.

### registry/

Responsible for model metadata, versions and references to stored model artifacts.

### utils/

Contains narrowly scoped shared utilities.

## 6. Application Architecture

The application will follow a layered architecture:

**Presentation layer:** Streamlit provides the user interface.

**API layer:** FastAPI provides validated endpoints for predictions, model information and analytical data.

**Core layer:** The `topicsense` package performs preprocessing, feature transformation, model loading, inference and evaluation.

**Persistence layer:** PostgreSQL stores application metadata and other explicitly approved persistent records.

**Experiment tracking layer:** MLflow tracks experiments, parameters, metrics and artifacts.

The frontend communicates with the backend through HTTP APIs. It must not directly access the database or duplicate ML inference logic.

The core NLP package must not depend on the frontend or API framework.

## 7. Database Architecture

PostgreSQL is the selected relational database.

Two logically separate databases will be maintained on the PostgreSQL server:

- `topicsense_app`: application data and metadata.
- `mlflow`: MLflow experiment-tracking metadata.

SQLAlchemy provides database access, Alembic manages schema migrations, and psycopg 3 provides PostgreSQL connectivity.

Model checkpoints and experiment artifacts must remain outside the relational metadata database.

Raw user-submitted documents will not be retained by default. Any decision to retain document content requires an explicit privacy and retention assessment.

## 8. MLflow Architecture

MLflow will be responsible for experiment tracking and model lifecycle management.

Tracked information will include:

- Experiment identifiers.
- Run identifiers.
- Model family and configuration.
- Hyperparameters.
- Dataset and split identifiers.
- Evaluation metrics.
- Training duration.
- Inference performance.
- Model artifact references.

MLflow metadata will use the dedicated PostgreSQL database. Artifacts will use a separate filesystem-backed storage location during local development, with persistent storage configured for Docker.

## 9. Development and Runtime Environments

### Local development

`uv` is the authoritative local Python dependency and environment manager.

Developers will use:

```bash
uv sync --all-groups
uv run python --version
```

Python commands should be executed using `uv run` to ensure the correct project environment is active.

### Docker

Docker provides isolated runtime environments for application services.

The Docker environment will:

- Use the project's Python version.
- Install dependencies from `pyproject.toml` and `uv.lock`.
- Preserve CPU-only PyTorch compatibility.
- Avoid copying the developer's local `.venv`.
- Exclude secrets and unnecessary local artifacts.
- Use persistent volumes for required database and tracking data.

The local `uv` environment and Docker environment share dependency definitions but remain separate runtime environments.

## 10. Testing Architecture

Testing will use pytest.

The testing structure separates:

- Unit tests.
- Integration tests.
- Prediction tests.

Testing must cover preprocessing, feature transformation, model interfaces, training behaviour, evaluation calculations, inference and application integration.

Expensive transformer training and large dataset downloads must not run as part of ordinary unit-test execution.

## 11. CI/CD Architecture

GitHub Actions will automate project quality checks.

The continuous integration pipeline will eventually include:

- Dependency and lockfile validation.
- Ruff linting.
- Ruff formatting validation.
- mypy static typing.
- pytest execution.
- Coverage reporting.
- Package and Docker build validation.

The release pipeline will build and publish versioned container images to GitHub Container Registry after successful validation.

Production deployment will only be introduced after the hosting environment, secrets, security requirements, migration strategy and rollback procedures have been defined.

## 12. Development Strategy

Development follows a phased, ML-first and integration-first approach.

The implementation order is:

1. Repository and architecture foundation.
2. Dataset engineering.
3. Text preprocessing and feature engineering.
4. Classical machine learning.
5. Model evaluation and experimental framework.
6. Neural network models.
7. Transformer experiments.
8. Comparative research analysis.
9. MLflow integration.
10. PostgreSQL persistence.
11. FastAPI backend.
12. Streamlit frontend.
13. Application integration.
14. Docker containerisation.
15. GitHub CI/CD.
16. Final validation and academic reporting.

Backend and frontend interfaces may be designed while ML development is underway, but implementation must use stable core interfaces.

## 13. Architecture Change Control

This architecture is the approved project baseline.

Any proposed structural or technology change must be reviewed before implementation.

The rationale, alternatives, trade-offs, consequences and migration requirements must be documented.

The repository architecture must not change silently.

## 14. Conclusion

TopicSense-NLP adopts a modular, reproducible and resource-aware architecture that integrates supervised NLP research with a practical application.

The design prioritises research integrity, maintainability, controlled experimentation and measurable system performance.