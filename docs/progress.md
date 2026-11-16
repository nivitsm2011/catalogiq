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
- Phase 3 commit is pushed to GitHub (`0d2fac8`).


## Phase 4 - Grounded listing generator (RAG, guardrails, evaluation)

**Done**
- Knowledge base: 800 Amazon listings (one document each, category group derived from the title because the Amazon
  `categories` field is empty) embedded with `all-MiniLM-L6-v2` into a persistent ChromaDB (`data/processed/chroma/`, 8.5 MB), plus
  13 style-guide chunks. `docs/style_guide.md` is generated from `configs/rag.yaml`, so the document and the checks cannot drift.
- Hybrid retrieval: in-house BM25 + vector search, reciprocal rank fusion, listing-quality boost, category filter.
- `LLMClient` interface (Ollama default, disk cache, timeouts), prompts in `configs/prompts/`, pydantic-validated listing JSON with
  one repair retry, guardrails (hallucination, banned words/claims, style, attribute coverage) and an optional guardrail-triggered rewrite.
- Evaluation on 50 held-out Kaggle test products, four arms, LLM-as-judge, owner's hand-check of 10 listings,
  `notebooks/04_listing_generator.ipynb`, `docs/model_card_listing_generator.md`. 20 new tests (LLM mocked); 56 tests in total, all passing.

**Metrics** (50 products, llama3.2:3b on CPU; guardrail numbers re-scored with the final checks)
| Arm | Unsupported-fact listings | Banned-word listings | Fully style-compliant | Judge clarity / accuracy | Sec per listing |
|---|---|---|---|---|---|
| no-RAG | 2% | 14% | 0% | 3.32 / 3.70 | 12.8 |
| RAG | 70% | 70% | 2% | 2.86 / 4.26 | 43.8 |
| no-RAG + rewrite | 0% | 2% | 0% | 3.44 / 3.68 | 35.9 |
| RAG + rewrite | 40% | 32% | 16% | 2.84 / 4.14 | 100.6 |

- Retrieval test (30 queries): precision@5 hybrid 0.78, vector 0.747, BM25 0.707; hit@3 vector 0.967, hybrid 0.933, BM25 0.90.
  Hybrid is the best all-rounder, not a dramatic winner; the quality boost costs about 0.02 precision@5.
- Retrieved examples hit the right category for 96% of products. Valid JSON 100% (repair used for 1 no-RAG listing).
- Hand-check (10 listings): judge vs human accuracy exact agreement 50%, Spearman 0.00; clarity 20%, -0.33. Not reliable.
- Footprint: embedder 90 MB (Hugging Face), Chroma 8.5 MB; project stays under 5 GB.

**Open issues / risks**
- **RAG with a 3B model copies facts from example listings** (leather, high-quality, breathable...). Next experiments: facts-stripped examples,
  fixed title template, a larger or hosted model through `LLMClient`.
- The 3B judge (same model as the writer) is not trustworthy on accuracy; the deterministic guardrails are the primary metric.
- Design was iterated (two metric bug fixes, extra claim words, the rewrite arm) while looking at the same 50 products: numbers are slightly optimistic.
- Hand-check is n=10; one item (6) rated 5 for accuracy contains three guardrail-flagged facts copied from an example.
- Seller notes in the evaluation are only the brand (from the catalog name); materials, sizes and weights were not supplied.
- Classifier colour is often unsure; 31 of 50 RAG titles still contain the unsure colour. The app should ask the seller to confirm attributes.
- The 100 duplicate labels from Phase 3 are still AI-provisional.
- Phase 4 commit is pushed to GitHub (`89e743a`).


## Phase 5 - Shopper Q&A over reviews (complete, pushed: `c16e398`)

**Done**
- Index: 139,400 reviews -> 166,607 chunks (short reviews whole, long ones in 3-sentence windows, <= 700 characters) with review id, product,
  rating, verified flag, helpful votes, date; own ChromaDB collection `amazon_reviews` (MiniLM, resumable build).
- Aspect tagging (six aspects) compared on 200 owner-labelled pieces: keyword rules, embedding similarity, zero-shot NLI (DeBERTa-v3-xsmall,
  142 MB download, approved), zero-shot LLM. Selection rule fixed in advance; rules used (see below).
- Pipeline `src/catalogiq/rag/qa.py`: per-product hybrid retrieval (BM25 + vector, RRF) + helpfulness/verified re-rank, evidence gate
  (>= 2 distinct reviews at similarity >= 0.52, else "Not enough reviews mention this." with no LLM call), JSON-validated answer with inline
  citations, agree/disagree lines, confidence, aspect, evidence_count; invalid ids stripped and counted.
- Product summaries (`summary.py`): numbers computed in code, LLM only phrases, per-aspect number check with template fallback.
- Evaluation on 60 questions / 20 products (45 answerable, 15 unanswerable), claim-level LLM judge, owner hand-check of 15 answers,
  `notebooks/05_review_qa.ipynb`, `docs/model_card_review_qa.md`, MLflow run `phase5_review_qa`. 23 tests in `tests/test_qa.py` (LLM mocked); 79 tests in total, all passing; ruff and black clean.

**Metrics** (llama3.2:3b on CPU)
| Metric | Value |
|---|---|
| Retrieval recall@5 (45 answerable): BM25 / vector / hybrid / hybrid + re-rank | 0.978 / 0.889 / 1.000 / 1.000 |
| MRR: hybrid / hybrid + re-rank | 0.829 / 0.893 |
| Correct refusal (unanswerable) | 93.3% (14 of 15) |
| False refusal (answerable) | 31.1% (14 of 45); first pipeline version 82% |
| Citations: invalid ids / relevance | 0 of 99 / 84.8% |
| LLM judge: fully faithful answers / claims supported | 40.6% of 32 / 56.7% (lower bound: judge too strict) |
| Hand-check (n=15): judge-human agreement / human faithful | 60% / 93% (6 of the judge's 7 flags overturned) |
| Evidence gate, cross-validated: answer rate / refuse rate | 60% / 82% |
| Aspect macro-F1 (100 test pieces): NLI / rules / embedding / LLM | 0.585 / 0.523 / 0.502 / 0.427 |
| NLI warm cost | 8.3 s per piece (about 16 days for the whole index) |
| Latency per question: answered / refused | 38.5 s / 1.9 s |
| Summaries (20 products): LLM / mixed / template | 18 / 1 / 1 |

**Open issues / risks**
- Evidence gate is a weak discriminator; its first dev/test tuning collapsed (test answer rate 45%) so it was cross-validated, and the final
  threshold 0.52 is fit on all 60 questions (end-to-end numbers slightly optimistic). Design was iterated on the same 60 questions.
- Faithfulness judge = same 3B model, too strict; hand-check is 15 answers by one person (14 faithful). A rough reweighted estimate is
  about 92% faithful but has no interval.
- NLI scored best on the pre-set rule but was not used (8.3 s per piece; gain +0.063 macro-F1, 95% CI [-0.007, 0.133]). Rules are used.
- The 200 aspect labels are liberal (quality/durability ticked on 144, 136 with 2+ aspects); delivery has only 14 positives.
- Failure Q27: the system said a pocket watch comes in a gift box; one cited review says "as a gift" (word match, not meaning) and the test
  label missed the other review's "fancy box". Next: require the question's key noun in a cited review; say "2 reviews" not "most".
- Answered questions take about 39 s on CPU: too slow for live use without caching, a faster model or a GPU.
- The Phase 3 duplicate labels are still AI-provisional.


## Phase 6 - Catalog quality checker (complete; commit `e153db0`)

**Done**
- Image checks with OpenCV (`quality/image_checks.py`): blur, darkness, washed-out exposure, resolution, aspect ratio, corner-based background clutter;
  text/watermark by CLIP zero-shot (`clip_checks.py`) after a blob heuristic failed. Thresholds tuned on validation (max recall at <= 3% of clean photos flagged).
- Degraded-copy generator (`degrade.py`): blur, dark, washed-out, shrunk, stretched (3 levels each), cluttered, text/watermark.
- Consistency (`consistency.py`): Phase 2 classifier AND CLIP zero-shot for colour, product type, gender; 300 injected mismatches per split, tuned on validation, reported on test.
- Completeness (`completeness.py`), 0-100 score with fixes (`score.py`), batch report over the 6,339 test listings (`scripts/run_quality_report.py` ->
  `reports/phase6/quality_report.csv` + `quality_summary.json`).
- `notebooks/06_quality_checker.ipynb`, `docs/model_card_quality_checker.md`, 13 new tests (`tests/test_quality.py`); 92 tests in total, all passing; ruff and black clean.

**Metrics** (test split)
| Metric | Value |
|---|---|
| Blurry: recall / precision / clean photos flagged | 0.999 / 0.42 / 2.4% |
| Too dark | 1.00 / 0.90 / 0.2% |
| Overexposed | 0.115 / 0.65 / 1.2% |
| Cluttered background | 0.45 / 0.75 / 1.0% |
| Text or watermark (CLIP; blob heuristic about 1% recall) | 0.60 / 0.39 / 2.6% |
| Low resolution / bad aspect ratio | 1.00 / 0.67 and 1.00 / 1.00 |
| Consistency, 300 injected: overall precision / recall | 0.47 / 0.48 (colour F1 0.23, type 0.53, gender 0.63) |
| Consistency, manual review of 24 untouched flagged listings | 12 likely catalog errors, 12 ambiguous, 0 clearly correct |
| Completeness: catalog / injected gaps found | 100% / 300 of 300 |
| Score: clean photo mean / AUC real-defect copies vs clean | 99.1 / 0.925 |
| Batch: listings with a flag / score below 85 / mean score | 8.3% / 4.1% / 98.7 |

**Open issues / risks**
- Jewellery (46.6% flagged: tiny items on white read as overexposed and blurry) and watches (14% text flags) are over-flagged; per-category thresholds are needed.
- Overexposure is nearly undetectable (recall 0.115); blur also fires on dark and washed-out photos (contrast dependence).
- All catalog photos are 60x80 studio shots: the phone-photo gap is unmeasured, defect recall comes from our own degraded copies, and only 2 of 120 natural photos had a defect.
- All Phase 6 labels are the AI assistant's (owner's choice), so they are provisional.
- Score weights are judgement, not fitted to outcomes. Colour mismatch detection is weak (F1 0.23).


## Phase 7 - Unified evaluation and scorecard (complete except the owner's timings; commit `26dae74`)

**Done**
- `src/catalogiq/eval/run_all.py` (`python -m catalogiq.eval.run_all`, `.\tasks.ps1 eval`, `make eval`): re-runs the Phase 2-6 evaluations on their fixed
  test sets, logs to MLflow experiment `catalogiq-eval` (one parent run tagged with the git commit hash and dirty flag, one nested run per component),
  checks drift against 75 numbers recorded in the earlier phases, measures end-to-end latency, peak RAM and disk, and writes `reports/scorecard.md`
  and `reports/phase7/{eval_results.json,drift.csv}`. Options: `--only`, `--resume`, `--reuse-image-eval`, `--skip-system`, `--render-only`.
- Phase 4 and 5 LLM steps are replayed through the on-disk LLM cache with a cache-miss budget (default 25), so a cold cache stops with a message instead of
  silently running for hours. Result: 0 LLM calls missed the cache in both phases.
- Scorecard sections: vision, visual search and duplicates, listing RAG vs no-RAG, review Q&A, quality checker, system, drift, product impact (every input
  labelled MEASURED or ASSUMPTION), limitations and risks (`docs/limitations_and_risks.md`), provenance.
- `scripts/make_timing_sheet.py` builds `reports/phase7/listing_timing.html`, a stopwatch page for the owner to time 5 manual listings (and optionally 5 draft reviews).
- 12 new tests (`tests/test_eval.py`); 104 tests in total, all passing; ruff and black clean.

**Results of the re-run** (the six evaluations take about 24 min, the uncached system test about 9 min)
| Item | Value |
|---|---|
| Numbers compared with earlier phases | 75: 68 match, 5 timing, 2 DRIFT (both timings, see below) |
| Vision mean macro-F1, fresh inference on 6,339 test photos | 0.771 (identical to Phase 2) |
| Phase 4 (4 arms x 50 products) and Phase 5 (60 questions), re-run through the LLM cache | every metric identical, 0 cache misses |
| Photo -> quality check -> attributes -> full listing, no-RAG | 15.5 s mean, 22.2 s p95 (uncached LLM, 5 photos) |
| Same, RAG + guardrail rewrite | 66.4 s mean, 78.1 s p95 |
| Peak RAM | 5.1 GB of 16.8 GB (1.6 GB Python with all models + 3.5 GB model server) |
| Disk | project folder 4.75 GB (budget 5 GB); with llama3.2:3b (2.0 GB, stored by Ollama) 6.75 GB |

**Open issues / risks**
- **The owner's timings of 5 manual listings are still missing.** Until `reports/phase7/manual_listing_times.csv` exists, every product-impact number is a
  labelled placeholder ASSUMPTION and "ILLUSTRATIVE". Time the listings with `reports/phase7/listing_timing.html`, then run `python -m catalogiq.eval.run_all --render-only`.
- The two DRIFT rows are timings: vector search 23.0 ms (Phase 3: 8.2) and CLIP encode 183 ms (60). A standalone check reproduced 188 ms, so it is the machine
  state, not the harness; Windows reported the laptop on battery power, which probably throttles the CPU (not verified). Re-measure plugged in.
- Project folder is 4.75 GB against the 5 GB budget, only 0.25 GB of headroom, and 6.75 GB with the language model. The Ollama store also holds two models
  CatalogIQ does not use (deepseek-r1:7b 4.7 GB, nomic-embed-text 0.3 GB).
- The harness caught two of my own bugs in the first run: a parsing error that made text-search drift (mAP 0.372 -> 0.903) and a RAM sampler that missed the
  `llama-server` process (peak RAM 2.3 GB before the fix, 5.1 GB after).
- Search and baselines are re-scored from cached embeddings and cached baseline predictions (recomputing the 42,257 CLIP embeddings takes about 2 hours).
- Phase 6 commit `e153db0` is not yet pushed to GitHub.


## Phase 8 - FastAPI backend and Streamlit app (complete; commit `8fca932`, pushed)

**Done**
- **API** (`api/`, docs in `docs/api.md`, interactive docs at `/docs`): `POST /predict-attributes`, `/generate-listing`, `/search/image`, `GET /search/text`,
  `POST /shop-the-look`, `/quality-check`, `/ask-reviews/{item_id}`, `GET /health`, `/metrics`, plus review products and aspect summaries, the ops summary and
  duplicate queue (with persisted decisions), demo products, dropdown options, the catalog photos and the scorecard. Pydantic schemas, photo validation (JPEG/PNG/WebP,
  at most 5 MB, real content checked), one error shape with readable messages, request logging with latency, in-memory request metrics, models loaded once at startup
  (about 1.5-2 minutes) with a background warm-up of the lazy parts, `503 starting_up` until ready.
- **App** (`app/`, run `.\tasks.ps1 app`): custom theme (`.streamlit/config.toml`), brand header and logo, animated KPI counters, Lottie loaders (three local animations made by
  `scripts/make_lottie.py`), animated bars and rings, Plotly bar charts with a Replay animation, cards and chips instead of raw tables. Pages: **Seller Studio** (photo or camera or
  demo product, editable attributes with confidence chips, listing with fact check, quality score with fix tips, similar listings, JSON/CSV download), **Shopper View**
  (text or photo search with filters, product detail, shop-the-look, aspect summary bars, review Q&A chat with clickable citations), **Catalog Ops** (KPIs, issue charts,
  worst listings with fixes, duplicate review queue with Merge / Keep both) and **Model Card** (the scorecard and an architecture diagram). Demo mode: 10 preloaded
  products whose listings and sample questions are replayed from the LLM cache, so they are instant.
- **Testing:** 25 API tests with `TestClient` and a fake state (`tests/test_api.py`); `scripts/e2e_smoke.py` drives a real headless Edge through every flow (12 checks, all pass,
  including a new listing generated live by the local LLM after editing an attribute); 7 screenshots in `docs/screenshots/` taken with `scripts/take_screenshots.py`.
  129 tests in total, all passing; ruff and black clean.

**Measured against the running server** (this CPU): attributes about 1.4 s, text or image search about 0.4 s, outfit 0.7 s, quality check about 1 s, cached listing about 1 s,
a new no-RAG listing about 15-30 s, a new answer about 40 s; startup 89-114 s.

**Open issues / risks**
- The photo catalog (Kaggle) and the reviews (Amazon) are different products, so the Shopper View shows review Q&A for a separate list of products (title only, no photo).
- New listings and answers are slow on CPU (seconds to a minute); the UI says so and demo products are instant. Real-time use would need a faster model or a GPU.
- Quality thresholds are tuned for 60x80 studio thumbnails; uploaded phone photos may behave differently (the page says so). The duplicate queue's threshold rests on AI-provisional labels.
- The built-in browser pane of the desktop app was hidden during testing, so the click-through ran in headless Edge (DevTools protocol) instead; the camera widget was not tested with real hardware.
- Streamlit and the API are single-user local servers: no authentication, no concurrency tuning, decisions in the duplicate queue are saved to a local JSON file.
- Phases 6, 7 and 8 are committed locally but not pushed to GitHub.


## Phase 9 - Hardening: tests, Docker, CI, robustness, performance

**Done**
- **Tests:** 129 -> 222 tests. Coverage of `src/catalogiq` + `api` rose from 62% to **75.5%**, measured in a clean checkout with no data, models or network
  (`pytest --cov`, fails below 70%). New fixtures in `tests/fixtures/` (18 KB). New unit tests for data cleaning, the review pipeline, prediction, aspect tagging, style guide,
  evaluation harness, config, utils and tracking; an end-to-end photo -> attributes -> listing -> quality test with a real tiny classifier and a mocked LLM; API tests for ops,
  scorecard, duplicates, limits and failures. Details: `docs/testing.md`.
- **CI:** `.github/workflows/ci.yml` runs ruff, `black --check` and pytest with coverage on every push (CPU PyTorch, no downloads except packages); badge in the README.
  **Not yet seen running on GitHub** (it runs on the first push); the same steps passed locally.
- **Docker:** `docker/Dockerfile.api`, `docker/Dockerfile.app` (multi-stage, slim, non-root, health checks), `docker-compose.yml` (api + app + ollama, volumes for data and models),
  `docs/docker.md` with commands and a disk-usage warning. **Written but not built: Docker is not installed on this laptop, so no image size was measured.**
- **Robustness:** Ollama not running or model missing -> `503 llm_unavailable` with instructions and a fallback panel in the app; startup check listing missing files with the fix;
  5.5 MB request cap, 429 rate limit (120 per minute, 20 for LLM endpoints), upload checks kept; all configured in `configs/api.yaml`.
- **Performance:** per-photo cache for CLIP and classifier results, startup warm-up, Ollama `keep_alive`. First photo flow after start 2.14 s -> 1.31 s; quality check 0.22 s -> 0.02 s;
  a new listing still takes about 36 s (LLM). Details and caveats: `reports/phase9/performance.md`.
- **Clean-up:** removed unused config keys and two unused functions; `docs/configuration.md` documents every config file, key group and environment variable.

**Open issues / risks**
- Docker images and the compose file are untested. CI has not run on GitHub yet; Linux differences (paths, process names) are possible.
- Not unit-tested on purpose: training and plotting scripts, the real CLIP encoder and the full-data evaluation components (the scorecard covers them).
- Performance numbers come from small samples on a laptop on battery. The language model is still the bottleneck for a new listing.
- The rate limiter is per process and in memory.

