# Progress

## Phase 0 - Project setup (complete, pushed)

- Windows 11 ARM64 (Snapdragon X), 15.6 GB RAM, no usable GPU (CPU only). Python 3.11.15 x64 venv via uv.
- Pinned `requirements.txt`, installable `catalogiq` package, config loader, logging, utils, task runner (`tasks.ps1`).
- Ollama 0.34.0 with `llama3.2:3b` (2.0 GB) answers a Python test prompt.
- Commits `e31bec5`, `5b92301`; pushed to https://github.com/nivitsm2011/catalogiq.

## Phase 1 - Data acquisition, cleaning and EDA (complete, pushed)

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
- x64 Python on ARM64 runs under emulation, so training will be slower than native.

## Phase 2 - Product attribute classifier (complete, committed locally, not yet pushed)

**Done**
- Baselines (majority class, frozen MobileNetV3 + logistic regression, zero-shot CLIP), all logged to MLflow.
- Multi-output MobileNetV3-Large (7 heads), two-stage training (4 heads-only + 6 fine-tune epochs, early stopping on
  validation macro-F1), class-weighted loss, hue-safe augmentation.
- Test evaluation, calibration (temperature scaling, ECE), per-target "not sure" thresholds, manual error analysis,
  Grad-CAM, `predict()` and TorchScript export, 20 passing tests (all suites), notebook `02_attribute_classifier.ipynb`,
  model card `docs/model_card_classifier.md`, 15 new decisions.

**Metrics (test split, 6,339 photos)**
| Target | Accuracy | Macro-F1 | Logreg macro-F1 | CLIP zero-shot macro-F1 |
|---|---|---|---|---|
| masterCategory | 0.992 | 0.990 | 0.976 | 0.883 |
| subCategory | 0.959 | 0.922 | 0.896 | 0.668 |
| articleType | 0.857 | 0.806 | 0.775 | 0.683 |
| baseColour | 0.675 | 0.582 | 0.469 | 0.561 |
| gender | 0.878 | 0.747 | 0.661 | 0.570 |
| season | 0.699 | 0.714 | 0.679 | 0.232 |
| usage | 0.876 | 0.639 | 0.622 | 0.387 |
| mean macro-F1 | | 0.771 | 0.725 | 0.569 |

- articleType top-3 accuracy 0.975; ECE 0.002-0.024 per target; CPU latency 18.7 ms mean / 22.2 ms p95 per image;
  TorchScript model 12.9 MB; training about 42 min CPU.
- Errors (articleType, 909 wrong): reviewed sample suggests about 10% plain label errors, 68% ambiguous boundaries,
  22% real model errors (one reviewer, n=40, indicative).

**Open issues / risks**
- Studio-to-phone photo domain gap still unmeasured (the catalog-quality-checker phase should address it).
- Colour (67.5% acc) and season remain weak; only 40% / 29% of photos clear the 90%-precision bar.
- First training run was killed mid fine-tune (cause unknown); resumed from the heads-only checkpoint.
- Test metrics come from one seed and one split; no confidence intervals.
- Model weights live in `models/` (gitignored); they must be regenerated from the scripts on a new machine.
- Phase 2 commit not yet pushed to GitHub (awaiting confirmation).
