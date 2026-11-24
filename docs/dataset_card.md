# TopicSense-NLP — Dataset Card

**Document status:** Initial template — Dataset selection pending  
**Version:** 1.0.0

## 1. Purpose

This document records the datasets used in TopicSense-NLP.

It provides a consistent record of dataset provenance, licensing, composition, preparation, intended use and limitations.

A separate dataset entry should be maintained for every source dataset.

## 2. Dataset Inventory

| Attribute | Description |
|---|---|
| Dataset name | Pending selection and verification |
| Dataset identifier | Pending |
| Source | Pending |
| Download location | Pending |
| Publication or release date | Pending |
| Download date | Pending |
| Dataset version | Pending |
| Licence | Pending verification |
| Language | Pending verification |
| Domain | Pending |
| Number of documents | Pending |
| Number of classes | Pending |
| File format | Pending |
| Checksum | Pending |

The proposed primary benchmark is 20 Newsgroups, subject to final verification.

Additional datasets will be selected after examining their suitability for supervised classification and cross-domain evaluation.

## 3. Dataset Selection Criteria

Datasets must satisfy the following requirements:

- Contain labelled textual documents.
- Have sufficiently documented provenance.
- Have an identifiable licence or usage policy.
- Provide suitable topic categories.
- Support reproducible acquisition.
- Be sufficiently representative of the intended domain.
- Have adequate class representation for evaluation.

## 4. Dataset Composition

For each dataset, document:

- Total document count.
- Class count.
- Per-class document count.
- Missing or empty documents.
- Duplicate documents.
- Document length distribution.
- Language distribution.
- Relevant metadata fields.

Class distributions should be visualised during dataset exploration.

## 5. Data Provenance and Licensing

For every dataset, record:

- Original publisher or maintainer.
- Official source.
- Version or release.
- Licence.
- Attribution requirements.
- Redistribution restrictions.
- Download date.
- Checksum, where available.

Do not assume that publicly accessible data is automatically licensed for redistribution.

## 6. Data Processing

The dataset preparation process may include:

- Schema standardisation.
- Text validation.
- Missing-value handling.
- Duplicate detection.
- Label validation.
- Text normalisation.
- Train/validation/test splitting.

Raw source data must remain unchanged.

Processed datasets must be traceable to their original sources and preparation configurations.

## 7. Data Splitting and Leakage Prevention

Splitting should be stratified where possible.

The test set must remain isolated from training and model selection.

Duplicate and near-duplicate records must be investigated before splitting.

Where appropriate, group-aware splitting should prevent documents from the same conversation, author or source group from leaking across evaluation partitions.

All split identifiers and random seeds must be recorded.

## 8. Cross-Domain Suitability

Datasets considered for cross-domain experiments must be assessed for:

- Domain differences.
- Label compatibility.
- Language consistency.
- Annotation consistency.
- Class distribution differences.

Label harmonisation must be justified semantically.

Datasets with incompatible labels must remain separate rather than being merged artificially.

## 9. Privacy and Ethical Considerations

Dataset processing must respect applicable licences and privacy requirements.

Sensitive personal information should not be collected or retained unnecessarily.

Social-media datasets require particular care regarding privacy, consent, platform terms and potential personal information.

## 10. Known Limitations

Limitations must be recorded after dataset inspection.

Potential issues include:

- Class imbalance.
- Domain-specific vocabulary.
- Outdated terminology.
- Noisy labels.
- Duplicate content.
- Limited language coverage.
- Sampling bias.
- Annotation inconsistencies.

No limitation should be claimed as observed until the dataset has been inspected.

## 11. Reproducibility Record

For each dataset, record:

| Field | Value |
|---|---|
| Dataset version | Pending |
| Source URL | Pending |
| Download date | Pending |
| Checksum | Pending |
| Preparation configuration | Pending |
| Split seed | Pending |
| Training document count | Pending |
| Validation document count | Pending |
| Test document count | Pending |

## 12. Dataset Status

**Current status:** Dataset selection and verification pending.

This document is a dataset governance template, not evidence that datasets have already been acquired, inspected or approved.