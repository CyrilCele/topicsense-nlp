# TopicSense-NLP

**A Reproducible, Multi-Domain NLP Research and Intelligence Platform for Supervised Text Classification**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20First-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-4169E1.svg)](https://www.postgresql.org/)
[![MLflow](https://img.shields.io/badge/Experiment%20Tracking-MLflow-0194E2.svg)](https://mlflow.org/)
[![CI/CD](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF.svg)](https://github.com/features/actions)
[![Status](https://img.shields.io/badge/Project-Research%20%26%20Development-orange.svg)](#project-status)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Project Overview

TopicSense-NLP is an academic research and software engineering project focused on the development of a reproducible, extensible and resource-aware natural language processing platform for supervised multi-class text classification.

The project investigates how different machine learning and deep learning approaches classify textual documents into predefined topic categories. It combines systematic experimentation with the engineering of a complete application, including a reusable NLP engine, experiment tracking, a REST API, a PostgreSQL-backed application layer and an interactive analytical dashboard.

The research considers multiple textual domains, including:

- News articles.
- Academic publications and abstracts.
- Social media text.
- Other suitable labelled textual collections.

The datasets will be selected and evaluated according to their suitability, label quality, licensing conditions, availability and compatibility. Cross-domain evaluation will be performed only where category mappings can be methodologically justified.

### 1.1 Central research focus

The project investigates the relationship between:

- Predictive classification performance.
- Model architecture and complexity.
- Text representation and preprocessing.
- Training dataset size.
- Cross-domain generalisation.
- Model confidence and uncertainty.
- Computational efficiency and resource requirements.

A particular engineering consideration is the development environment: an Intel Core i3-1115G4 processor, two physical CPU cores, four logical processors and approximately 11 GB of memory allocated to WSL2. GPU acceleration is unavailable.

Rather than treating these constraints as an obstacle to rigorous research, the project incorporates resource consumption and computational efficiency into its experimental methodology.

### 1.2 Terminological clarification

Although the assignment uses the expression *topic modeling*, the principal implementation addresses **supervised topic classification**.

Supervised classification learns from labelled documents and predicts predefined categories. It is distinct from unsupervised topic modelling approaches such as Latent Dirichlet Allocation (LDA).

The primary research deliverable will therefore be a supervised multi-class text classification system. Unsupervised topic modelling may be discussed as related background but will not replace the required supervised methodology.

---

## 2. Project Vision and Deliverables

TopicSense has two complementary deliverables.

### 2.1 Academic research deliverable

A structured and reproducible investigation covering:

1. Dataset acquisition, provenance and quality assessment.
2. Text preprocessing and feature representation.
3. Classical machine learning baselines.
4. Neural-network-based text classification.
5. Transformer-based classification.
6. Controlled model and hyperparameter comparisons.
7. Training dataset size experiments.
8. Cross-domain evaluation.
9. Predictive performance and computational efficiency.
10. Error analysis, limitations and research conclusions.

### 2.2 Software engineering deliverable

A functional NLP application consisting of:

- A modular Python NLP engine.
- Reusable data processing and model training components.
- Experiment tracking through MLflow.
- A PostgreSQL-backed application database.
- A FastAPI REST backend.
- A Streamlit frontend.
- Interactive visualisations through Plotly.
- Model metadata and version management.
- Automated testing and quality checks.
- Docker-based local infrastructure.
- GitHub Actions continuous integration and delivery workflows.

The application will expose trained models through a consistent inference interface, allowing users to submit text, classify documents and inspect relevant model information and evaluation results.

---

## 3. Research Objectives

The project aims to:

1. Identify, acquire and document appropriate labelled text datasets.
2. Investigate dataset quality, class distributions, linguistic characteristics and potential biases.
3. Develop model-appropriate text preprocessing and representation pipelines.
4. Establish reproducible classical machine learning baselines.
5. Investigate neural architectures for text classification.
6. Evaluate pretrained transformer-based classification.
7. Compare predictive performance across model families.
8. Investigate the influence of training dataset size.
9. Evaluate generalisation across compatible textual domains.
10. Investigate model confidence, uncertainty and calibration where feasible.
11. Measure training time, inference performance and computational requirements.
12. Develop a reusable application and model-serving architecture.
13. Establish reproducible experiment tracking and software quality controls.
14. Produce an academically defensible research report and complete technical documentation.

---

## 4. Research and Modelling Strategy

### 4.1 Classical machine learning

The classical machine learning experiments will investigate:

- Multinomial Naive Bayes.
- Logistic Regression.
- Linear Support Vector Machine (Linear SVM).

Text representation will primarily use:

- Bag-of-Words.
- Term Frequency–Inverse Document Frequency (TF-IDF).
- Unigram and bigram features.

Potential preprocessing experiments include:

- Case normalisation.
- Tokenisation.
- Stopword handling.
- Stemming.
- Lemmatization.
- Punctuation and numerical token handling.
- Vocabulary filtering.
- N-gram configuration.

Preprocessing choices will be evaluated rather than assumed to be universally beneficial.

### 4.2 Neural-network-based classification

The deep-learning component will investigate:

- TextCNN.
- Bidirectional LSTM (BiLSTM), subject to computational feasibility.

PyTorch will provide the neural-network implementation framework.

These models will be evaluated against the classical baselines using documented data partitions and comparable evaluation procedures.

### 4.3 Transformer-based classification

The principal transformer experiment will investigate fine-tuning:

- DistilBERT using PyTorch and Hugging Face Transformers.

A secondary experiment may investigate frozen sentence embeddings using a compact pretrained model such as MiniLM, followed by a lightweight classifier.

Transformer preprocessing will respect the tokenizer and input requirements of the selected pretrained model. Operations such as stemming, lemmatization and indiscriminate stopword removal will not automatically be applied to transformer inputs.

### 4.4 Model comparison

The project will compare three broad modelling families:

| Family | Planned approaches |
|---|---|
| Classical ML | Naive Bayes, Logistic Regression, Linear SVM |
| Neural networks | TextCNN, optional BiLSTM |
| Transformers | DistilBERT, optional frozen MiniLM embeddings |

The best model will not be selected solely because it is the most complex architecture.

Conclusions will be based on measured predictive performance, generalisation, computational efficiency and suitability for the intended application.

---

## 5. Dataset Strategy

### 5.1 Primary benchmark

20 Newsgroups is the proposed primary benchmark because it provides a manageable starting point for multi-class text classification.

The final selection and configuration will be confirmed during dataset acquisition and validation.

### 5.2 Cross-domain evaluation

Additional labelled datasets representing news, academic text and social media will be investigated.

These datasets will be treated as separate evaluation sources where their topic definitions differ.

Dataset merging will not be performed merely to increase the number of documents. Category compatibility and label semantics must first be examined and documented.

### 5.3 Dataset acceptance criteria

Every dataset will be assessed according to:

- Source and provenance.
- Licensing and redistribution conditions.
- Document count.
- Label definitions.
- Class distribution.
- Language and text quality.
- Missing or invalid records.
- Duplicate and near-duplicate documents.
- Annotation quality and potential noise.
- Suitability for the intended experiments.

### 5.4 Data organisation

The project will maintain the following data separation:

| Directory | Purpose |
|---|---|
| `data/raw/` | Original acquired datasets |
| `data/interim/` | Intermediate preparation outputs |
| `data/processed/` | Validated modelling datasets |
| External storage | Resources that should not be committed to Git |

Raw datasets will be preserved without modification wherever licensing and storage arrangements permit.

Dataset versions, acquisition procedures and transformations will be documented.

---

## 6. Experimental Design and Evaluation

### 6.1 Data partitioning

Experiments will use training, validation and test partitions.

Where appropriate official benchmark splits exist, they will be preserved.

The validation set will support model selection and hyperparameter optimisation. The final test set will remain independent of model development.

Learned preprocessing operations, vocabulary construction and feature extraction will be fitted exclusively on training data.

Where document relationships, threads or near-duplicates create leakage risks, appropriate group-based partitioning will be considered.

### 6.2 Evaluation metrics

Classification evaluation will include:

- Accuracy.
- Precision.
- Recall.
- Macro F1-score.
- Weighted F1-score.
- Per-class precision, recall and F1-score.
- Confusion matrices.

Computational evaluation will include:

- Training duration.
- Inference latency.
- Inference throughput.
- Model artifact size.
- Memory consumption where measurable.
- Dataset and training sample counts.

Macro F1-score will be the primary comparative metric where class imbalance makes accuracy insufficient.

Additional analyses may include:

- Confidence calibration.
- Reliability diagrams.
- Uncertainty and abstention behaviour.
- Learning curves.
- Cross-domain performance.
- Statistical variability.
- Model error analysis.

### 6.3 Training dataset size experiments

The project will investigate progressively increasing training data sizes:

- 10%.
- 25%.
- 50%.
- 75%.
- 100%.

Validation and test partitions will remain fixed to support meaningful comparisons.

The sampling procedure will preserve class representation wherever possible.

### 6.4 Hyperparameter optimisation

Optuna will be investigated for systematic hyperparameter optimisation.

Search spaces will be appropriate to individual model families and the available computational resources.

Optimisation will use validation performance, never final test-set results.

### 6.5 Resource-aware experimental policy

The project will prioritise:

- Small-scale feasibility experiments before expensive training.
- CPU-compatible batch sizes.
- Appropriate sequence-length limits.
- Dynamic padding where supported.
- Checkpointing.
- Controlled parallel execution.
- Recorded training and inference durations.
- Memory monitoring.
- Documented deviations from the planned experimental protocol.

A computationally expensive model will not automatically be considered superior to a lighter alternative.

---

## 7. Technology Stack

| Area | Technology |
|---|---|
| Programming language | Python 3.12 |
| Environment and dependency management | uv |
| Data manipulation | NumPy, Pandas |
| Classical machine learning | scikit-learn |
| Deep learning | PyTorch |
| Pretrained NLP models | Hugging Face Transformers |
| Dataset utilities | Hugging Face Datasets, where appropriate |
| Hyperparameter optimisation | Optuna |
| Experiment tracking | MLflow |
| Backend | FastAPI |
| Frontend | Streamlit |
| Interactive visualisation | Plotly |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Database migrations | Alembic |
| PostgreSQL driver | psycopg |
| Research environment | JupyterLab |
| Static visualisation | Matplotlib, Seaborn |
| Testing | pytest |
| Code quality | Ruff, mypy |
| API testing | HTTPX / FastAPI TestClient |
| Version control | Git and GitHub |
| Large-file management | Git LFS, selectively |
| Containerisation | Docker and Docker Compose |
| CI/CD | GitHub Actions |
| Security analysis | GitHub security tooling, where available |

Dependency versions will be managed through `pyproject.toml` and `uv.lock`.

The PyTorch CPU package source will be retained in the dependency configuration to support the CPU-only development environment.

Additional dependencies will only be introduced when they support an approved project requirement.

---

## 8. Frozen Repository Architecture

The following repository structure is the **approved and frozen target architecture** for TopicSense-NLP.

It establishes the project's architectural boundaries and responsibilities. New functionality must be implemented within the appropriate existing layer.

Structural changes require documented review and an Architecture Decision Record (ADR). The architecture is not to be casually reorganised during implementation.

```text
topicsense-nlp/
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── cd.yml
│   │   └── codeql.yml
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
│
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE
├── .gitignore
├── .dockerignore
├── .python-version
├── .env.example
│
├── configs/
│   ├── data.yaml
│   ├── classical_ml.yaml
│   ├── neural_networks.yaml
│   ├── transformers.yaml
│   ├── experiments.yaml
│   └── application.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── README.md
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
│       ├── experiments/
│       ├── explainability/
│       ├── inference/
│       ├── registry/
│       └── utils/
│
├── backend/
│   └── topicsense_api/
│       ├── __init__.py
│       ├── main.py
│       ├── api/
│       │   └── v1/
│       ├── schemas/
│       ├── services/
│       ├── dependencies/
│       └── database/
│           ├── session.py
│           ├── base.py
│           └── models/
│
├── frontend/
│   ├── app.py
│   ├── pages/
│   ├── components/
│   └── assets/
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
│   ├── api/
│   └── test_prediction.py
│
├── database/
│   ├── migrations/
│   │   └── versions/
│   └── README.md
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── tracking/
│   └── .gitkeep
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
    ├── adr/
    ├── architecture.md
    ├── methodology.md
    ├── experiments.md
    ├── dataset_card.md
    └── model_card.md
```

### 8.1 Architectural responsibilities

| Component | Responsibility |
|---|---|
| `src/topicsense/` | Core NLP functionality and reusable business logic |
| `backend/` | REST API, request validation and application services |
| `frontend/` | User interface and interactive dashboard |
| `configs/` | Externalised configuration |
| `notebooks/` | Research exploration and analysis |
| `scripts/` | Repeatable operational entry points |
| `tests/` | Automated quality assurance |
| `database/` | Database migrations and documentation |
| `docker/` | Container infrastructure |
| `tracking/` | Local MLflow tracking and supporting configuration |
| `models/` | Model checkpoints and registry artifacts |
| `results/` | Generated experimental results |
| `reports/` | Technical and academic reports |
| `docs/` | Architecture, methodology and governance documentation |

### 8.2 Frozen architectural principles

The following principles are mandatory:

1. Core NLP logic must reside in `src/topicsense/`.
2. The backend must call the core package rather than duplicate its functionality.
3. The frontend must communicate with backend services through defined interfaces.
4. Database access must be isolated within the application persistence layer.
5. Notebooks must import reusable functionality rather than become independent implementations.
6. Scripts must remain thin execution interfaces.
7. Tests must cover important functionality and integration boundaries.
8. Configuration must be externalised where appropriate.
9. Models and generated artifacts must remain separate from source code.
10. Structural changes must be reviewed and documented.

The directory structure is frozen as the target. Directories will be created progressively when their corresponding implementation phases begin.

---

## 9. Application Architecture

### 9.1 High-level architecture

```text
                  USER
                    │
                    ▼
          STREAMLIT FRONTEND
                    │
              HTTP / REST
                    │
                    ▼
             FASTAPI BACKEND
                    │
          ┌─────────┼─────────┐
          │         │         │
          ▼         ▼         ▼
      NLP ENGINE  DATABASE  MLFLOW
          │         │         │
          ▼         ▼         ▼
      ML MODELS  PostgreSQL  Tracking
          │
          ▼
      PREDICTIONS
          │
          ▼
       RESULTS
```

The architecture separates research functionality, application services, persistence, experiment tracking and presentation.

### 9.2 FastAPI backend

The backend will provide versioned REST endpoints, initially including:

| Endpoint | Responsibility |
|---|---|
| `GET /api/v1/health` | Application health |
| `POST /api/v1/predict` | Single-document classification |
| `POST /api/v1/predict/batch` | Batch classification |
| `GET /api/v1/models` | Available models |
| `GET /api/v1/models/{id}` | Model metadata |
| `GET /api/v1/metrics` | Available evaluation summaries |

The final endpoint contracts will be documented and tested.

The backend will handle request validation, error responses, model access and application persistence.

### 9.3 Streamlit frontend

The dashboard will provide:

- Project overview.
- Single-text classification.
- Batch document classification.
- Model selection and model information.
- Model comparison.
- Experiment analytics.
- Dataset exploration.
- Classification error analysis.
- Explainability visualisations.
- Model registry views.
- Application status and monitoring.

Plotly will support interactive analytical visualisations.

All performance figures displayed as research results must originate from recorded experiments.

### 9.4 PostgreSQL

PostgreSQL is the selected relational database technology.

The initial local architecture will separate application data from MLflow tracking data using two databases:

- `topicsense_app`
- `mlflow`

The application database may contain prediction history, model metadata, application records and other approved persistent information.

MLflow will use its own tracking backend database.

Model checkpoints, large datasets and MLflow artifacts will be stored separately from relational database records.

SQLAlchemy will provide database access and Alembic will manage schema migrations.

Credentials and connection strings must be supplied through environment variables, never hardcoded or committed.

---

## 10. MLflow and Experiment Management

MLflow will provide experiment tracking and model lifecycle support.

The experiment tracking implementation will record, where applicable:

- Experiment identifiers.
- Run identifiers.
- Model architecture.
- Dataset version.
- Training and validation configurations.
- Hyperparameters.
- Random seeds.
- Evaluation metrics.
- Training duration.
- Inference measurements.
- Model artifacts.
- Execution status and relevant notes.

MLflow will complement rather than replace the project's research and experiment registers.

The project will maintain its own conventions for experiment identifiers, configurations, dataset provenance and result interpretation.

Local MLflow artifact storage will be used initially. Remote artifact storage may be introduced if deployment requirements justify it.

---

## 11. Reproducibility and Data Governance

Reproducibility is a core engineering and research requirement.

The project will enforce:

- Fixed and documented random seeds.
- Versioned dependencies and committed lockfiles.
- Explicit dataset provenance.
- Versioned experiment configurations.
- Consistent evaluation protocols.
- Separation of training, validation and test data.
- Train-only fitting of learned preprocessing.
- Model and dataset metadata.
- Recorded computational measurements.
- Automated tests and quality checks.

### 11.1 Dataset and artifact governance

Raw datasets, processed datasets, model checkpoints and experiment outputs will be managed according to their licensing, storage and reproducibility requirements.

Git LFS may be used for suitable large files, but it will not be used to bypass dataset licensing or repository storage limitations.

Generated artifacts will not automatically be committed to Git.

### 11.2 Security

The project will not commit:

- API keys.
- Database passwords.
- Private environment files.
- Access tokens.
- Sensitive user-submitted data.
- Unauthorised datasets.

A `.env.example` file will document required configuration variables without containing real credentials.

---

## 12. Testing and Quality Assurance

Automated quality assurance will be integrated into the development lifecycle.

### 12.1 Unit tests

Unit tests will validate individual components, including:

- Data validation.
- Preprocessing.
- Feature extraction.
- Configuration loading.
- Model interfaces.
- Evaluation metrics.
- Prediction validation.
- Database utilities.

### 12.2 Integration tests

Integration tests will verify:

- Data loading and preparation.
- Training and evaluation workflows.
- Model saving and loading.
- API-to-NLP-engine integration.
- Database persistence.
- MLflow tracking integration.

### 12.3 API tests

The REST API will be tested for:

- Valid requests.
- Invalid requests.
- Empty text.
- Unsupported inputs.
- Model availability.
- Expected response structures.
- Error handling.

### 12.4 Code quality

Ruff will enforce linting and formatting standards.

Mypy will support static type checking.

Pytest will execute the automated test suite.

---

## 13. GitHub Actions: Continuous Integration and Continuous Delivery

GitHub Actions will automate quality checks and delivery workflows.

The project will use three principal workflow definitions.

### 13.1 Continuous Integration — `ci.yml`

CI will run on relevant pushes and pull requests.

The workflow will progressively include:

1. Checkout repository.
2. Set up Python 3.12.
3. Install uv.
4. Synchronise dependencies from the lockfile.
5. Run Ruff lint checks.
6. Run formatting checks.
7. Run mypy.
8. Execute pytest.
9. Generate coverage information.
10. Build the Python package.
11. Validate relevant application and API components.

Pull requests must pass the required checks before they can be merged into the protected main branch.

Expensive transformer training and full research experiments will not run as ordinary CI jobs.

CI tests will use small, deterministic fixtures and lightweight test models.

### 13.2 Continuous Delivery — `cd.yml`

The CD workflow will support controlled release delivery.

The intended process is:

- Trigger on approved version tags or release events.
- Run the required quality gates.
- Build the application container image.
- Publish versioned images to GitHub Container Registry (GHCR).
- Associate release artifacts with the relevant version.
- Support a protected, environment-gated deployment stage when a deployment target has been selected.

Production deployment will not be assumed or automatically enabled before hosting infrastructure, secrets, security requirements and operational procedures have been established.

### 13.3 Security analysis — `codeql.yml`

GitHub security analysis will be incorporated where supported by the repository and account configuration.

Security checks will include appropriate static analysis and dependency vulnerability review.

Dependabot may be configured for dependency update proposals.

### 13.4 Branching and release policy

The intended Git workflow is:

- `main`: stable and protected integration branch.
- `feature/*`: new functionality.
- `fix/*`: defect corrections.
- `docs/*`: documentation changes.
- `experiment/*`: isolated research implementation when required.

Pull requests will provide an auditable review point.

Releases will use semantic versioning where appropriate:

- MAJOR: incompatible application or interface changes.
- MINOR: backward-compatible functionality.
- PATCH: backward-compatible corrections.

The GitHub Actions workflows will be implemented progressively and documented as part of the software delivery process.

---

## 14. Docker and Local Infrastructure

Docker and Docker Compose will be used to standardise local application infrastructure.

The intended containerised services are:

- PostgreSQL.
- FastAPI backend.
- Streamlit frontend.
- MLflow tracking service.

Development services may be started selectively rather than all at once.

The local environment will use persistent volumes where appropriate.

The container architecture will account for the available CPU and memory resources. Model training will not be forced into a container configuration that creates unnecessary resource contention.

Production deployment configuration will be developed only after the application has passed its integration and performance requirements.

---

## 15. Resource-Aware Development Policy

The development environment has the following recorded specifications:

| Resource | Specification |
|---|---|
| Operating environment | Ubuntu on WSL2 |
| Processor | 11th Gen Intel Core i3-1115G4 |
| Physical cores | 2 |
| Logical processors | 4 |
| Memory allocated to WSL2 | Approximately 11 GB |
| GPU acceleration | Unavailable |
| Python | 3.12 |
| PyTorch | CPU build |

The project will prioritise efficient experimentation.

Initial transformer feasibility settings will be conservative and adjusted through measurement.

Potential controls include:

- Sequence length.
- Batch size.
- Gradient accumulation.
- Number of training epochs.
- Dynamic padding.
- CPU thread allocation.
- Checkpoint frequency.
- Training subset size.
- Concurrent service usage.

These settings are experimental starting points, not universal fixed values.

The final report will document hardware constraints, experimental compromises, runtime measurements and their implications for result interpretation.

---

## 16. Documentation Standards

The project will maintain the following documentation:

| Document | Purpose |
|---|---|
| `README.md` | Project overview, setup and architecture |
| `docs/architecture.md` | Detailed system architecture |
| `docs/methodology.md` | Research methodology |
| `docs/experiments.md` | Experiment design and tracking conventions |
| `docs/dataset_card.md` | Dataset provenance and characteristics |
| `docs/model_card.md` | Model configurations, intended use and limitations |
| `docs/adr/` | Architectural Decision Records |
| `data/README.md` | Dataset acquisition and organisation |
| `notebooks/README.md` | Notebook execution and research conventions |
| `database/README.md` | PostgreSQL setup and migration instructions |

The academic report will be based on actual experimental evidence and will follow the applicable university writing and referencing guidelines.

---

## 17. Development Roadmap

The project will be developed through controlled work packages.

| Work package | Deliverable | Status |
|---|---|---|
| WP01 | Research blueprint and frozen architecture | Planned |
| WP02 | Dataset acquisition and validation | Planned |
| WP03 | Classical ML pipeline | Planned |
| WP04 | Neural-network experiments | Planned |
| WP05 | Transformer experiments | Planned |
| WP06 | Advanced evaluation and error analysis | Planned |
| WP07 | MLflow experiment tracking | Planned |
| WP08 | PostgreSQL persistence | Planned |
| WP09 | FastAPI backend | Planned |
| WP10 | Streamlit dashboard | Planned |
| WP11 | Docker and application integration | Planned |
| WP12 | GitHub CI/CD and release engineering | Planned |
| WP13 | Final research report and project audit | Planned |

The roadmap may be refined as implementation evidence becomes available, but the approved architectural boundaries will remain in force unless formally revised.

---

## 18. Project Status

**Current phase:** Initial project setup and architectural planning.

| Component | Status |
|---|---|
| WSL2 development environment | Completed |
| Python environment | Completed |
| CPU-only PyTorch installation | Completed |
| Initial repository setup | Completed |
| Research direction | Established |
| Target architecture | Approved |
| Dataset acquisition | Pending |
| Data validation | Pending |
| Classical ML implementation | Pending |
| Neural-network implementation | Pending |
| Transformer implementation | Pending |
| MLflow integration | Pending |
| PostgreSQL integration | Pending |
| FastAPI backend | Pending |
| Streamlit frontend | Pending |
| Docker infrastructure | Pending |
| GitHub CI/CD workflows | Pending |
| Final research report | Pending |

Project status will be updated to reflect actual implementation and verification, not merely planned work.

---

## 19. Architectural Governance and README Review

This README defines the approved project baseline and frozen target repository architecture.

The architecture will be governed through documented decisions.

Any proposed structural change must:

1. Identify the problem with the existing architecture.
2. Explain why the current design is insufficient.
3. Consider reasonable alternatives.
4. Assess implementation and maintenance implications.
5. Record the selected decision in an ADR.
6. Update relevant architecture documentation.

This policy allows necessary engineering improvements without uncontrolled architectural drift.

### Final project review

At the end of the project, this README will be revisited and audited against the implementation.

The final review will verify:

- Whether all documented components were implemented.
- Whether the repository matches the approved architecture.
- Whether the research objectives were achieved.
- Whether the application works as documented.
- Whether CI/CD workflows operate correctly.
- Whether all dependencies and setup instructions are accurate.
- Whether reported results can be reproduced.
- Whether known limitations and deviations are disclosed.

The README will then be updated to represent the actual delivered system rather than the original plan.

---

## 20. License

This project is intended to use the MIT License, subject to the presence and contents of the repository's `LICENSE` file.

The license applies to original project code and documentation. It does not automatically grant rights to external datasets, pretrained models, libraries or third-party resources.

All third-party materials remain subject to their respective licensing terms and usage restrictions.

---

## 21. Author

**Nkululeko Cyril Cele**

- GitHub: [@CyrilCele](https://github.com/CyrilCele)
- Project: TopicSense-NLP

TopicSense-NLP is an academic and professional portfolio project exploring supervised natural language processing, classical machine learning, deep learning, transformer architectures, reproducible experimentation and full-stack ML application engineering.

---

**TopicSense-NLP — Research rigor. Reproducible engineering. Measurable intelligence.**