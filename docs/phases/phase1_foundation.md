# Phase 1: Foundation

**Status:** IN PROGRESS
**Estimated time:** ~1 hour

## Tasks

- [x] Init git repo
- [x] Create directory structure
- [x] pyproject.toml with all dependencies
- [x] .gitignore
- [x] Makefile with all targets
- [x] src/ package with __init__.py files
- [x] config.py (Settings with pydantic-settings)
- [x] README.md skeleton
- [x] HIGH_LEVEL_PLAN.md in repo
- [ ] Download dataset from VeritaResearch/claim-extraction
- [ ] Write src/data/dataset.py (load, preprocess, split)
- [ ] Verify data loading works
- [ ] Initial git commit

## Dataset Details

Source: https://github.com/VeritaResearch/claim-extraction
Format: CSV with columns `text` and `label` (0 = not claim, 1 = claim)
Split: 80/20 train/test (pre-split in the repo, or we split ourselves)

## Deliverable

```bash
python -c "from src.data.dataset import load_claim_dataset; ds = load_claim_dataset(); print(ds)"
```
Should print dataset info with ~10,400 train / ~2,600 test samples.
