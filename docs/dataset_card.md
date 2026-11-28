# TopicSense-NLP — Dataset Card

**Dataset:** 20 Newsgroups  
**Internal identifier:** `20newsgroups`  
**Project version:** 1.0.0

## 1. Dataset Overview

TopicSense-NLP uses the 20 Newsgroups dataset as its supervised multi-class text classification benchmark.

The dataset contains historical Usenet discussion documents grouped into 20 newsgroup categories.

Each document is associated with a predefined category label.

The dataset is used to investigate text classification, document representation, model comparison and classification errors.

## 2. Dataset Characteristics

| Characteristic | Value |
|---|---:|
| Total documents | 18,846 |
| Number of classes | 20 |
| Original training documents | 11,314 |
| Original test documents | 7,532 |
| Development training documents | 9,051 |
| Validation documents | 2,263 |
| Language | English |
| Data modality | Text |
| Classification type | Supervised multi-class |

The development training and validation partitions originate from the original training partition.

The official test partition remains separate from model selection.

## 3. Source and Acquisition

The dataset is acquired using the scikit-learn `fetch_20newsgroups` interface.

Original dataset information:

- Original dataset source: https://people.csail.mit.edu/jrennie/20Newsgroups/
- scikit-learn dataset documentation: https://scikit-learn.org/stable/datasets/real_world.html

The acquisition and validation procedures are implemented through the project's data-ingestion functionality.

## 4. Topic Categories

The dataset contains the following 20 categories:

1. `alt.atheism`
2. `comp.graphics`
3. `comp.os.ms-windows.misc`
4. `comp.sys.ibm.pc.hardware`
5. `comp.sys.mac.hardware`
6. `comp.windows.x`
7. `misc.forsale`
8. `rec.autos`
9. `rec.motorcycles`
10. `rec.sport.baseball`
11. `rec.sport.hockey`
12. `sci.crypt`
13. `sci.electronics`
14. `sci.med`
15. `sci.space`
16. `soc.religion.christian`
17. `talk.politics.guns`
18. `talk.politics.mideast`
19. `talk.politics.misc`
20. `talk.religion.misc`

## 5. Data Organisation

The repository organises the dataset into three processing stages:

| Directory | Purpose |
|---|---|
| `data/raw/` | Acquired source data |
| `data/interim/` | Intermediate processing outputs |
| `data/processed/` | Prepared data and partition information |

Processed split information is maintained in:

`data/processed/20newsgroups/splits/`

The split manifests include:

- `official_training_split.json`
- `decontaminated_training_split.json`
- `preprocessing_split_manifest.json`

These files support consistent partition handling throughout the research workflow.

## 6. Data Splitting Strategy

The original dataset provides separate training and test partitions.

The original training partition is divided into:

| Partition | Documents |
|---|---:|
| Development training | 9,051 |
| Validation | 2,263 |
| Total | 11,314 |

The official test partition contains 7,532 documents.

The development training partition is used for model fitting. The validation partition supports model comparison and error analysis.

The official test partition is kept separate from model selection.

## 7. Data Quality and Integrity

Dataset preparation considers:

- Missing or invalid document content.
- Document duplication.
- Label validity.
- Class distribution.
- Data partition integrity.
- Potential overlap and contamination between partitions.

Duplicate and contamination analysis supports the identification of potential data leakage.

The project maintains split manifests to support consistent use of the prepared partitions.

## 8. Text Representation

The experiments investigate different representations of document text.

These include:

- Full-message text.
- Message-body text.
- Bag-of-Words features.
- TF-IDF features.
- Transformer-based representations.
- Sentence Transformer embeddings.

Different representations are evaluated within the corresponding modelling experiments.

## 9. Intended Uses

The dataset supports research into:

- Supervised text classification.
- Multi-class document categorisation.
- Classical machine-learning baselines.
- Neural text classification.
- Transformer-based text classification.
- Text representation comparisons.
- Classification error analysis.

## 10. Limitations and Considerations

The 20 Newsgroups dataset has several limitations:

- It represents historical Usenet discussions rather than contemporary online communication.
- Its topic categories do not represent all possible document classification domains.
- Some documents contain headers, signatures and quoted material that can influence classification.
- Historical vocabulary and communication conventions may differ from modern text.
- The dataset may contain duplicated or closely related documents.
- Performance on this dataset does not establish generalisation to unrelated datasets or domains.

Results should therefore be interpreted within the context of the dataset and experimental design.

## 11. Ethical Considerations

The dataset contains historical user-generated text.

Although it is used for academic and technical research, its content should be treated responsibly.

Classification results should not be interpreted as reliable assessments of individual authors, their identities or their personal characteristics.

## 12. References

- Lang, K. (1995). NewsWeeder: Learning to filter netnews. *Proceedings of the 12th International Conference on Machine Learning*.
- scikit-learn documentation: https://scikit-learn.org/stable/datasets/real_world.html
- Original 20 Newsgroups dataset: https://people.csail.mit.edu/jrennie/20Newsgroups/

## 13. Summary

The 20 Newsgroups dataset provides the common experimental foundation for TopicSense-NLP.

Its documented partitions, preprocessing procedures and quality assessments support consistent and reproducible supervised multi-class text classification experiments.