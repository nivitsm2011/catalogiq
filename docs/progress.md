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

## Phase 2 - Product attribute classifier (complete, pushed)

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

## Phase 3 - Visual search and duplicate detection (complete except for the owner's own duplicate labels)

**Done**
- OpenCLIP ViT-B/32 (laion2b, 605.2 MB) embeddings for all 42,257 photos and attribute sentences, cached with an id map
  (`data/processed/clip_*.npy`); Phase 2 classifier features cached as a baseline.
- Exact (`IndexFlatIP`) vs HNSW comparison; `SearchEngine` with `search_by_image`, `search_by_text`, `search_by_id` and
  exact attribute filters; `ShopTheLook` (style-family rules, colour/usage/season scoring, classifier label gate, reasons);
  `DuplicateFinder` (CLIP similarity + pHash); retrieval metrics; click-to-label page for 100 duplicate pairs.
- Notebook `03_visual_search.ipynb` (8 example queries, 2 explained failures, text search, outfits, duplicates, t-SNE map),
  `docs/search_design.md`, 11 new decisions, 16 search tests (36 tests in total, all passing; ruff and black clean).

**Metrics**
| Image search (1,995 test queries vs 35,918-photo gallery; relevant = same type AND colour) | Recall@5 | Recall@10 | Precision@10 | mAP@10 |
|---|---|---|---|---|
| CLIP ViT-B/32 (zero-shot) | 0.823 | 0.890 | 0.418 | 0.309 |
| Phase 2 fine-tuned classifier features | 0.772 | 0.854 | 0.412 | 0.312 |
| Frozen ImageNet MobileNetV3 features | 0.698 | 0.796 | 0.319 | 0.223 |
| Random vectors (floor) | 0.040 | 0.088 | 0.010 | 0.003 |

- Text to image (300 attribute queries): Recall@10 0.887, Precision@10 0.451, mAP@10 0.372 (random: 0.017 / 0.002 / 0.0003).
- Example grid (8 queries x top 5): 36 of 40 results are the right type, 19 also the right colour.
- Index at 35,918 items: exact 5.1 ms/query; HNSW (efSearch 64) 0.18 ms at 99.9% of the exact top-10. Synthetic 718k items:
  exact 87 ms vs HNSW 0.57 ms. Switch point on this CPU about 400k items (50 ms budget).
- Latency over the 42k catalog: vector search 8.2 ms (13.9 ms filtered); CLIP encode 60 ms per image, 40 ms per text.
- Duplicates (**provisional AI labels**, 38 duplicate / 58 not / 4 unsure): threshold 0.975 + pHash <= 8, all 27 sampled pairs
  flagged are duplicates; weighted recall 0.22 (very uncertain); catalog-wide 1,833 pairs, 2,143 products (5.1%) in 875 groups.
- Leakage check: 71 of 6,339 test photos (1.1%) have a re-shot twin in train/val; Phase 2 articleType accuracy 85.7% -> 85.5%
  without them.
- Footprint: data 1.7 GB, models 0.64 GB, venv about 1.7 GB: about 4.1 GB, under the 5 GB limit.

**Open issues / risks**
- The duplicate labels are the AI assistant's, not the owner's. Replace them: label `reports/duplicate_labeling.html`, save the
  CSV as `reports/duplicate_labels.csv`, then run `python scripts/tune_duplicates.py` and re-execute notebook 03.
- Duplicate recall below 0.95 similarity rests on 4 labelled duplicates standing for about 5,900 pairs: treat as unknown.
- Cosmetics shade variants (lipstick, nail polish) can be flagged as duplicates; add a shade check before auto-merging.
- Shop-the-look has no ground truth and untuned weights; the classifier gate rarely lets Tunics, Flats and Kurtis through.
- CLIP embedding needed about 2 hours of CPU (and the laptop slept overnight); a catalog refresh should embed only new items.
- Phase 3 commit not yet pushed to GitHub (awaiting confirmation).
