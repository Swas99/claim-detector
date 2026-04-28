# Dataset

The composite dataset is downloaded from [VeritaResearch/claim-extraction](https://github.com/VeritaResearch/claim-extraction).

Run `make download-data` or `python -m src.data.dataset --download` to fetch it.

## Sources

| Source | Records | % Claims |
|---|---|---|
| Claimbuster (Hassan et al., 2017) | 7,976 | 25.0% |
| PoliClaim Gold (Ni et al., 2024) | 1,953 | 59.1% |
| AVeriTeC (Schlichtkrull et al., 2023) | 3,068 | 100.0% |
| **Total** | **12,997** | **47.8%** |

Split: 80% train (~10,400), 20% test (~2,600).
