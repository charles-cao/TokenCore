# TokenCore

Official code and datasets for the paper
**[Towards Token-Level Text Anomaly Detection](https://huggingface.co/papers/2601.13644)** (WWW '26).

TokenCore is a memory-bank method for **token-level text anomaly detection**: instead of only flagging whole documents as anomalous, it localizes *which specific tokens* are anomalous. It builds a memory bank of normal token embeddings and scores each test token by its distance to the nearest normal neighbor, then aggregates token scores to obtain document-level scores.

## Datasets

The three benchmark datasets are hosted on the Hugging Face Hub:

👉 **[https://huggingface.co/datasets/charles-cao/TokenAD](https://huggingface.co/datasets/Charles-Cao/TokenAD)**

Each dataset provides both document-level and token-level anomaly annotations.

| Dataset | Anomaly type | Documents | Anomalies | Rate |
|---|---|---|---|---|
| SMS Spam | Character-level (corrupted sequences) | 4,518 | 393 | 8.7% |
| Restaurant Review | Semantic (negative sentiment) | 1,100 | 50 | 4.5% |
| Grammar Correction | Grammatical errors | 300 | 30 | 10% |

A local copy is also available in the `data/` folder.

## Installation

```bash
git clone https://github.com/charles-cao/TokenCore.git
cd TokenCore
pip install -r requirements.txt
```

## Quick Start

```python
from TokenCore import TextCore

model = TokenCore()

# Fit on normal token embeddings
model.fit(X_train)

# Get token-level anomaly scores
scores = model.decision_function(X_test)
```

`X_train` and `X_test` are token embedding matrices (e.g. BERT embeddings with max-pooling over subwords). See `Demo.ipynb` for the full pipeline, including embedding extraction, token-level scoring, and document-level aggregation.

## Repository Structure

```
TokenCore/
├── data/           # Benchmark datasets (also on Hugging Face Hub)
├── TextCore.py     # TokenCore method implementation
├── Demo.ipynb      # End-to-end demo of the key steps
└── README.md
```

## Citation

```bibtex
@inproceedings{cao2026tokenlevel,
  title     = {Towards Token-Level Text Anomaly Detection},
  author    = {Cao, Yang and Yu, Bicheng and Yang, Sikun and Liu, Ming and Yang, Yujiu},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year      = {2026},
  doi       = {10.1145/3774904.3792952}
}
```

## Links

- Paper (Hugging Face): https://huggingface.co/papers/2601.13644
- Paper (ACM): https://dl.acm.org/doi/abs/10.1145/3774904.3792952
- Datasets: https://huggingface.co/datasets/charles-cao/TokenCore
