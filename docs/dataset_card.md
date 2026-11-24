# TopicSense-NLP — Dataset Card

**Document status:** Primary benchmark selected; local acquisition pending  
**Version:** 1.1.0  
**Last updated:** 24 November 2026

## 1. Dataset Overview

The primary benchmark selected for TopicSense-NLP is the **20 Newsgroups dataset**.

It is a collection of Usenet discussion documents organised into 20 newsgroup categories. It is suitable for supervised multi-class text classification.

The dataset is distributed with an original training partition and an original test partition.

## 2. Dataset Identification

| Attribute | Value |
|---|---|
| Dataset name | 20 Newsgroups |
| Internal identifier | `20newsgroups` |
| Original source | https://people.csail.mit.edu/jrennie/20Newsgroups/ |
| Acquisition interface | `sklearn.datasets.fetch_20newsgroups` |
| Feature type | Text |
| Language | English |
| Number of classes | 20 |
| Total documented samples | 18,846 |
| Original training samples | 11,314 |
| Original test samples | 7,532 |
| Domain | Usenet discussion posts |
| Original split | Date-based |
| Local format | JSON Lines |
| Redistribution licence | Requires verification against original source |

Dataset-level counts and partition details are documented by scikit-learn:
https://scikit-learn.org/stable/datasets/real_world.html

## 3. Topic Categories

The dataset contains the following categories:

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

## 4. Acquisition Strategy

The dataset will be acquired using the scikit-learn dataset loader.

The original training and test partitions will be preserved.

The acquisition configuration uses:

- `remove=()` to retain the original document content.
- `shuffle=False` to preserve deterministic source ordering.
- The original `train` and `test` subsets.
- A dedicated local cache directory.

No text cleaning, tokenisation, stopword removal or feature extraction will occur during acquisition.

## 5. Local Storage

The intended storage layout is:

```text
data/
└── raw/
    └── 20newsgroups/
        ├── cache/
        ├── 20newsgroups_raw.jsonl
        └── 20newsgroups_manifest.json
```

The raw JSONL file stores:

- Record identifier.
- Dataset identifier.
- Original source partition.
- Original document text.
- Category name.
- Numeric category identifier.
- SHA-256 content hash.

The manifest records dataset metadata, class distribution, source partition counts, retrieval timestamp, software version and raw file checksum.

## 6. Licensing and Redistribution

The dataset is publicly obtainable through the documented source and scikit-learn loader.

However, the existence of a public download does not establish unrestricted redistribution rights for the original Usenet posts.

The redistribution licence and applicable original terms must be verified before redistributing the raw corpus.

Until then:

- Raw documents remain outside Git.
- Raw documents are not included in published project archives.
- Dataset source attribution is retained.
- Redistribution restrictions are treated as unresolved.

## 7. Data Quality Assessment

The following checks will be performed after acquisition:

- Total document count.
- Category count.
- Per-category distribution.
- Empty or whitespace-only documents.
- Missing labels.
- Invalid records.
- Exact duplicate documents.
- Duplicate documents across original partitions.
- Document length distribution.
- Encoding and text integrity.

Observed values will be recorded after the validation process has been executed.

## 8. Metadata and Potential Leakage

The original documents may contain:

- Newsgroup headers.
- Author or sender information.
- Signatures.
- Quoted messages.
- Email addresses and other metadata.

Some metadata may reveal the target category without representing the actual topic of the document.

The original text will therefore be preserved during acquisition.

Later experiments will compare controlled text representations, including metadata-preserving and metadata-reduced variants.

The results must distinguish genuine topic classification performance from performance that may rely on metadata.

## 9. Splitting Strategy

The original dataset provides a training partition and a test partition.

The project will preserve the original test partition for final evaluation.

The original training partition will subsequently be divided into training and validation subsets using stratified splitting, subject to class representation.

The validation set will support model selection and hyperparameter optimisation.

The original test set must not be used during training or model selection.

Duplicate and near-duplicate analysis must be completed before finalising the evaluation protocol.

## 10. Cross-Domain Research

20 Newsgroups will serve as the primary benchmark.

Additional datasets will be evaluated separately for cross-domain experiments.

Candidate datasets must have:

- Documented provenance.
- A verifiable licence or usage policy.
- Labelled textual records.
- Sufficient class representation.
- A defensible relationship between their topic categories and the primary benchmark.

Datasets will not be merged merely to increase sample count.

## 11. Ethical and Privacy Considerations

The corpus consists of historical discussion posts.

The project will avoid unnecessary exposure of personal information contained in headers, signatures or document bodies.

Raw text will not be exposed through the application by default.

## 12. Acquisition Record

| Field | Recorded value |
|---|---|
| Acquisition status | Pending execution |
| Acquisition date | Pending |
| scikit-learn version | Pending |
| Actual document count | Pending |
| Actual class count | Pending |
| Training partition count | Pending |
| Test partition count | Pending |
| Raw file SHA-256 | Pending |
| Exact duplicate count | Pending |
| Licence verification | Pending |

## 13. Limitations

Known dataset characteristics include historical Usenet content, domain-specific vocabulary and the potential for metadata leakage.

Additional limitations will be documented after inspecting the downloaded corpus.

The dataset is a benchmark, not a representative sample of all modern textual domains.

## 14. Dataset Status

**Selected:** Yes  
**Acquired locally:** Pending  
**Validated:** Pending  
**Approved for redistribution:** No — licence verification pending  
**Approved for model training:** Pending validation and split preparation