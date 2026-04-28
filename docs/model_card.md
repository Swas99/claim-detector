# Model Card: Claim Detection (DistilBERT)

## Model Details

- **Model**: DistilBERT-base-uncased, fine-tuned for binary classification
- **Task**: Sentence-level claim detection (is this a verifiable factual claim?)
- **Parameters**: 66 million
- **Training**: 5 epochs, lr=2e-5, batch_size=16, Apple Silicon MPS
- **Framework**: HuggingFace Transformers + PyTorch

## Intended Use

- Fact-checking triage: identify sentences worth sending to fact-checkers
- Content moderation: flag factual claims in user-generated content
- Research: automated claim extraction from political speeches, news, social media

This model is NOT a fact-checker. It identifies whether a claim CAN be checked, not whether it is true or false.

## Training Data

Composite dataset from three human-curated sources (~13,000 sentences):

| Source | Records | % Claims | Domain |
|---|---|---|---|
| Claimbuster | 7,976 | 25.0% | US political debates |
| PoliClaim Gold | 1,953 | 59.1% | Political speeches |
| AVeriTeC | 3,068 | 100.0% | Fact-check articles |

80/20 train/test split. Dataset from [VeritaResearch/claim-extraction](https://github.com/VeritaResearch/claim-extraction).

## Performance

### In-domain (test set, 2,600 samples)

| Metric | Score |
|---|---|
| Accuracy | 91.1% |
| Precision | 89.6% |
| Recall | 91.5% |
| F1 | 0.905 |

### Model Comparison

| Model | F1 | Notes |
|---|---|---|
| TF-IDF + XGBoost | 0.797 | Classical baseline |
| **DistilBERT** | **0.905** | **Deployed model** |
| BERT-base | 0.905 | Paper reproduction |
| ModernBERT | 0.917 | Best overall |

### Confidence Calibration

Post-hoc temperature scaling (T=2.15) applied to improve confidence reliability. NLL improved by 40.9% after calibration.

## Limitations

- **Domain bias**: Trained primarily on political speech and fact-check data. May underperform on other domains (scientific papers, social media, legal text).
- **Out-of-domain**: The reference paper shows BERT-family models drop to ~0.77 F1 on tweets (CheckThat dataset). LLMs generalize better out-of-domain.
- **Language**: English only. No multilingual support.
- **Sentence-level**: Designed for single sentences. Does not handle document-level claim extraction or multi-sentence reasoning.
- **Not fact-checking**: Cannot determine if a claim is true or false, only if it is checkable.

## Ethical Considerations

- **Bias**: Training data skews toward US political content. Claims from non-Western contexts may be misclassified.
- **Misuse**: Could be used to selectively flag claims from specific political viewpoints. Should be deployed with human oversight.
- **Confidence**: Calibrated confidence scores are more reliable than raw softmax, but should not be treated as ground truth for automated decisions.
