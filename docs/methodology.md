# TopicSense-NLP — Research Methodology

**Document status:** Approved baseline  
**Version:** 1.0.0

## 1. Research Objective

The primary objective is to develop and evaluate a supervised NLP system for multi-class topic classification across different textual domains.

The research compares classical machine-learning algorithms, neural network architectures and transformer-based models under controlled experimental conditions.

The investigation considers both predictive performance and computational efficiency.

## 2. Research Questions

The project investigates the following questions:

**RQ1:** How effectively can supervised machine-learning models classify documents into predefined topic categories?

**RQ2:** How does training dataset size influence classification performance?

**RQ3:** Which classical machine-learning algorithm provides the strongest classification performance?

**RQ4:** How do text preprocessing and feature representation choices influence classification performance?

**RQ5:** How effectively do trained models generalise across different textual domains?

**RQ6:** Do neural network architectures provide meaningful improvements over classical machine-learning approaches?

**RQ7:** Does transformer fine-tuning justify its computational cost in a CPU-constrained environment?

**RQ8:** How do the confidence and calibration characteristics of different model families compare?

**RQ9:** Which model provides the most appropriate balance between predictive performance and computational efficiency?

## 3. Research Design

The project uses a quantitative experimental research design.

Models will be trained using labelled datasets and evaluated using held-out data.

Experiments will compare model families under controlled conditions, with consistent data splits and documented preprocessing configurations.

The primary evaluation task is supervised multi-class classification.

## 4. Dataset Strategy

The project will use a hybrid dataset strategy consisting of:

- A primary benchmark dataset.
- Additional labelled datasets for cross-domain evaluation.

The proposed primary benchmark is the 20 Newsgroups dataset, subject to final source, licence and suitability verification.

Additional datasets may include news articles, academic abstracts and social-media text.

Datasets will not be merged automatically. Label harmonisation will only occur where semantic equivalence can be justified.

Otherwise, cross-domain evaluation will be performed using separate dataset-specific label spaces and clearly documented transfer settings.

## 5. Data Quality and Preparation

Each dataset must be inspected before model training.

The inspection process includes:

- Source and licence verification.
- Dataset size.
- Class distribution.
- Missing values.
- Duplicate records.
- Language consistency.
- Document length distribution.
- Label consistency.
- Potential data leakage.

The dataset must be standardised into a consistent internal representation containing document text, label and relevant metadata.

Dataset-specific metadata must be retained where useful for analysis.

## 6. Data Splitting

The preferred evaluation strategy is stratified splitting where class distributions permit it.

The dataset will be divided into:

- Training set.
- Validation set.
- Test set.

The test set must remain untouched during model selection and hyperparameter optimisation.

Preprocessing transformations that learn from data, including vocabulary construction and TF-IDF weighting, must be fitted only on the training data.

Where documents share authors, conversations, threads or near-duplicate content, group-aware splitting will be considered to reduce leakage.

Random seeds and split identifiers must be recorded.

## 7. Text Preprocessing

The preprocessing investigation will consider:

- Text normalisation.
- Case handling.
- Tokenisation.
- Stopword removal.
- Stemming.
- Lemmatisation.
- Punctuation handling.
- URL and HTML processing.
- Vocabulary filtering.

Preprocessing decisions must be justified experimentally.

Stopword removal will not be applied indiscriminately because some words may contribute meaningful topic information.

The same preprocessing assumptions must be applied consistently when comparing relevant model configurations.

## 8. Feature Representation

Classical machine-learning models will use numerical text representations, primarily:

- Bag of Words.
- TF-IDF.
- Unigram features.
- Unigram and bigram combinations.

Feature extraction must be fitted using training data only.

Neural and transformer models will use tokenised or embedded representations appropriate to their architectures.

## 9. Model Families

### 9.1 Classical Machine Learning

The mandatory classical baselines are:

- Multinomial Naive Bayes.
- Logistic Regression.
- Linear Support Vector Machine.

These models provide computationally efficient baselines.

### 9.2 Neural Networks

The planned neural architectures are:

- TextCNN.
- Bidirectional LSTM, if computationally feasible.

### 9.3 Transformer Models

The primary transformer candidate is DistilBERT.

A frozen sentence-transformer embedding approach may be considered as an additional experiment if resources and time permit.

Transformer experiments will use bounded CPU configurations and documented resource measurements.

## 10. Training and Hyperparameter Optimisation

Each model must be trained using documented configurations.

Hyperparameter optimisation will use Optuna selectively.

Optimisation must use training and validation data only.

The final test set must not be used for hyperparameter selection.

All relevant seeds, model parameters and runtime settings must be recorded.

## 11. Evaluation Metrics

The evaluation framework will report:

- Accuracy.
- Precision.
- Recall.
- Macro F1-score.
- Weighted F1-score.
- Per-class precision, recall and F1-score.
- Confusion matrix.

Computational evaluation will additionally consider:

- Training duration.
- Inference latency.
- Inference throughput.
- Peak memory usage where measurable.
- Model artifact size.

Where probability estimates are available and meaningful, calibration will be investigated using appropriate methods.

Macro F1-score will be a primary comparison metric because it provides equal weight to each class.

Accuracy and weighted F1-score will provide complementary information.

## 12. Learning Curve Analysis

The influence of training dataset size will be investigated using progressively larger training subsets.

Proposed fractions:

- 10%.
- 25%.
- 50%.
- 75%.
- 100%.

Validation and test sets will remain fixed during these experiments.

Sampling will preserve class representation where possible.

## 13. Cross-Domain Evaluation

Cross-domain experiments will investigate how models trained on one domain perform on another.

The source and target datasets must have compatible labels or an explicitly justified label-mapping strategy.

Results will distinguish between:

- In-domain performance.
- Cross-domain performance.
- Domain adaptation, if implemented.

A decline in performance will be analysed rather than hidden.

## 14. Reproducibility

Experiments must record:

- Dataset identity and version.
- Data split identifiers.
- Random seeds.
- Preprocessing configuration.
- Feature representation.
- Model architecture.
- Hyperparameters.
- Software versions.
- Hardware configuration.
- Training and inference runtime.
- Evaluation metrics.

MLflow will be used to track experiments once its integration phase is implemented.

## 15. Computational Constraints

The project is developed on CPU-only hardware.

Classical machine-learning models are the mandatory first priority.

Neural and transformer experiments will be introduced progressively after successful smoke tests.

Experiment configurations must account for available memory, CPU threads, sequence length, batch size and training duration.

Computational infeasibility must be reported honestly.

## 16. Research Integrity

No performance result may be invented or presented without an actual experiment.

The final report must distinguish measured results from expectations and hypotheses.

All datasets must have documented provenance and licensing.

Potential data leakage, class imbalance, domain mismatch and limitations must be explicitly discussed.

## 17. Expected Research Contribution

The expected contribution is a reproducible comparative investigation of supervised multi-class topic classification across multiple textual domains, with particular attention to the relationship between predictive performance, training data size, model complexity and computational efficiency.