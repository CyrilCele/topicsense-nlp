# TopicSense-NLP — Model Card

## 1. Model Overview

**Project:** TopicSense-NLP  
**Task:** Supervised multi-class text classification  
**Dataset:** 20 Newsgroups  
**Number of classes:** 20  
**Project version:** 1.0.0

TopicSense-NLP investigates multiple model families for assigning documents to predefined newsgroup categories.

The project evaluates classical machine-learning algorithms, neural networks and transformer-based approaches.

This model card documents the evaluated model families and the strongest recorded validation configuration.

## 2. Intended Task

The classification task is to predict one of 20 predefined topic categories from the text of a newsgroup document.

The task is supervised because model training uses documents associated with known category labels.

The project investigates:

- Document classification.
- Multi-class text categorisation.
- Model comparison.
- Text representation.
- Classification error analysis.

## 3. Evaluated Model Families

### 3.1 Classical Machine Learning

The classical models include:

- Multinomial Naive Bayes.
- Logistic Regression.
- Linear Support Vector Machine.

The experiments use Bag-of-Words and TF-IDF representations.

Both full-message and message-body text are considered.

### 3.2 Neural Networks

The neural architectures include:

- TextCNN.
- Bidirectional LSTM (BiLSTM).

These approaches investigate learned neural representations for document classification.

### 3.3 Transformer-Based Models

The transformer approaches include:

- Fine-tuned DistilBERT.
- Frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings with Logistic Regression.

The first approach adapts a pretrained transformer to the classification task.

The second approach uses fixed sentence embeddings as input to a separate classifier.

## 4. Selected Validation Configuration

The strongest recorded validation configuration is:

**Linear Support Vector Machine with TF-IDF features using full-message text.**

Its recorded validation metrics are:

| Metric | Validation result |
|---|---:|
| Accuracy | 91.47% |
| Macro F1-score | 0.9138 |

These figures are validation results.

They must not be represented as official test-set performance.

## 5. Training and Evaluation Data

The dataset partitions are:

| Partition | Documents | Purpose |
|---|---:|---|
| Development training | 9,051 | Model fitting |
| Validation | 2,263 | Model comparison and error analysis |
| Official test | 7,532 | Separate held-out evaluation |

The official test partition is kept separate from model selection.

## 6. Evaluation Metrics

The project evaluates classification performance using:

- Accuracy.
- Precision.
- Recall.
- Macro F1-score.
- Confusion matrices.
- Class-level classification reports.

Macro F1-score provides a class-balanced perspective by calculating performance across categories without weighting them by their support.

## 7. Intended Use

The evaluated models are intended for research and educational investigation of supervised text classification.

Potential applications within the scope of the research include:

- Categorising documents by topic.
- Comparing classification algorithms.
- Investigating text representations.
- Studying model errors.
- Evaluating multi-class classification approaches.

The reported performance applies to the documented dataset and experimental conditions.

## 8. Limitations

The models have several important limitations:

- They are evaluated on the 20 Newsgroups dataset.
- The dataset consists of historical Usenet discussions.
- The models may not generalise to contemporary communication or unrelated domains.
- Classification performance can be influenced by document headers, quoted text and signatures.
- Similar documents and duplicated content can affect evaluation if data partitions are not properly controlled.
- Aggregate metrics do not fully describe class-specific errors.
- Validation performance does not establish independent official test-set performance.

## 9. Ethical Considerations

The models classify documents according to topic categories.

They are not designed to identify authors, infer personal characteristics or make decisions about individuals.

Predictions should be interpreted as automated topic classifications rather than definitive assessments of document meaning or author intent.

## 10. Reproducibility

Reproducibility is supported by:

- Version-controlled source code.
- Configuration files.
- Documented dataset partitions.
- The `uv.lock` dependency lockfile.
- Sequential research notebooks.
- Reusable model implementations.
- Consistent evaluation procedures.

## 11. Appropriate Interpretation

The validation results demonstrate performance under the documented experimental conditions.

They should not be interpreted as evidence of universal text-classification performance.

Independent evaluation on additional datasets would be necessary before making claims about generalisation beyond the 20 Newsgroups benchmark.

## 12. Conclusion

TopicSense-NLP provides a comparative investigation of classical machine learning, neural networks and transformer-based approaches to supervised multi-class text classification.

The model card documents the evaluated approaches, intended task, validation results, limitations and appropriate interpretation of the findings.