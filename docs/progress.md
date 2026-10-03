# Progress

## Phase 0 - Project setup (complete, pushed)

- Windows 11 ARM64 (Snapdragon X), 15.6 GB RAM, no usable GPU (CPU only). Python 3.11.15 x64 venv via uv.
- Pinned `requirements.txt`, installable `catalogiq` package, config loader, logging, utils, task runner (`tasks.ps1`).
- Ollama 0.34.0 with `llama3.2:3b` (2.0 GB) answers a Python test prompt.
- Commits `e31bec5`, `5b92301`; pushed to https://github.com/nivitsm2011/catalogiq.

## Phase 1 - Data acquisition, cleaning and EDA (complete, committed locally)

**Done**
- Fashion Product Images (Small): downloaded 592.6 MB, audited, cleaned, split (see `docs/data_card.md`).
- Amazon_Fashion: streamed 2.5M reviews / 825,869 items, sampled 800 items and 139,400 cleaned reviews (17.5 MB).
- Listing quality score and exemplar flag; keyword stats for listings and reviews.
- Narrated EDA notebook `notebooks/01_data_understanding.ipynb` with 5 figures in `reports/figures/`.
- Docs: `docs/data_card.md`, 13 new rows in `docs/decisions.md`.

**Metrics (all computed)**
| Item | Value |
|---|---|
| styles.csv rows / repaired / lost | 44,446 / 22 / 0 |
| Missing images / corrupt images | 5 / 0 |
| Cleaned rows | 42,257 |
| Classes after cleaning (master / sub / articleType / colour / gender / season / usage) | 4 / 25 / 55 / 15 / 5 / 4 / 6 |
| Near-duplicate groups (pHash <= 1) | 1,553 groups, 3,932 images, largest 30 |
| Split sizes train / val / test | 29,579 / 6,339 / 6,339 (70.0 / 15.0 / 15.0%) |
| Groups spanning splits | 0 |
| Amazon sample | 800 items, 139,400 reviews, 17.5 MB (cap 300 MB), 191 exemplar listings |
| Twin-photo label disagreement (master / sub / type / colour / gender / season / usage) | 0.5 / 3.8 / 12.0 / 44.0 / 7.9 / 18.1 / 9.9 % |
| Unit tests | 11 passing; ruff and black clean |
| Disk: `data/` | about 0.7 GB (project total about 2.4 GB incl. `.venv`, excl. Ollama models) |

**Open issues / risks**
- Studio-photo domain gap vs small-seller phone photos is unmeasured (Phase 2 / quality checker).
- Season and usage are provisional targets; Phase 2 decides on measured lift over the majority baseline.
- Colour labels are noisy (44% twin disagreement); evaluate colour on the 15-family palette.
- Phase 1 commit not yet pushed to GitHub (awaiting confirmation).
- x64 Python on ARM64 runs under emulation, so training will be slower than native.
