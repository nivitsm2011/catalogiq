# CatalogIQ

[![CI](https://github.com/nivitsm2011/catalogiq/actions/workflows/ci.yml/badge.svg)](https://github.com/nivitsm2011/catalogiq/actions/workflows/ci.yml)

An AI copilot for e-commerce marketplaces: it helps small sellers create better listings, helps shoppers
get grounded answers, and helps catalog teams monitor quality. Runs fully locally on CPU.

> Status: Phase 9 complete (tests, Docker, CI, hardening). Full README coming as features land.

See [CLAUDE.md](CLAUDE.md) for the project brief, [docs/progress.md](docs/progress.md) for status and
[docs/decisions.md](docs/decisions.md) for design decisions.

## Quick start (Windows PowerShell)

```powershell
.\tasks.ps1 setup   # create .venv, install pinned requirements, install the package
.\tasks.ps1 test
.\tasks.ps1 lint
```

## Data pipeline (Phase 1)

Data is stored locally only and is gitignored. Windows PowerShell:

```powershell
kaggle datasets download paramaggarwal/fashion-product-images-small -p data/raw/fashion --unzip   # 593 MB
.\.venv\Scripts\python.exe scripts/build_fashion.py     # audit, clean, group-aware 70/15/15 splits
.\.venv\Scripts\python.exe scripts/build_reviews.py     # streams Amazon_Fashion, writes ~18 MB sample
```

Kaggle needs a token in `~/.kaggle/access_token`. See [docs/data_card.md](docs/data_card.md) and
[notebooks/01_data_understanding.ipynb](notebooks/01_data_understanding.ipynb).

## Attribute classifier (Phase 2)

```powershell
.\.venv\Scripts\python.exe scripts/run_baselines.py          # baselines (downloads CLIP weights, 605 MB, once)
.\.venv\Scripts\python.exe scripts/train_classifier.py --epochs1 4 --epochs2 6 --tag full   # about 42 min on CPU
.\.venv\Scripts\python.exe scripts/evaluate_classifier.py --tag full
.\.venv\Scripts\python.exe scripts/export_model.py --tag full
```

```python
from catalogiq.vision.predict import predict
predict("some_photo.jpg")   # {attribute: (label, confidence, needs_review)}
```

Test results, error analysis and limitations: [docs/model_card_classifier.md](docs/model_card_classifier.md) and
[notebooks/02_attribute_classifier.ipynb](notebooks/02_attribute_classifier.ipynb). Trained weights are not in git.

## Visual search and duplicates (Phase 3)

```powershell
.\.venv\Scripts\python.exe scripts/embed_catalog.py        # CLIP embeddings for all photos (about 2 h on CPU, resumable)
.\.venv\Scripts\python.exe scripts/compare_indexes.py      # exact vs HNSW
.\.venv\Scripts\python.exe scripts/evaluate_search.py      # Recall@k, mAP, text search, latency
.\.venv\Scripts\python.exe scripts/make_labeling_sheet.py  # 100 duplicate pairs to label (open the HTML page)
.\.venv\Scripts\python.exe scripts/tune_duplicates.py      # pick the duplicate threshold from the labels
```

```python
from catalogiq.search.embeddings import ClipEncoder, embed_catalog
from catalogiq.search.engine import SearchEngine
emb, _, catalog = embed_catalog()
engine = SearchEngine(catalog, emb, encoder=ClipEncoder())
engine.search_by_text("black formal shoes for men", k=5)
engine.search_by_image(pil_image, k=5, filters={"gender": "Women"})
```

Design, outfit rules and evaluation protocol: [docs/search_design.md](docs/search_design.md);
results and failure cases: [notebooks/03_visual_search.ipynb](notebooks/03_visual_search.ipynb).

## Grounded listing generator (Phase 4)

```powershell
.\.venv\Scripts\python.exe scripts/build_knowledge_base.py   # style guide, brand lexicon, ChromaDB (downloads MiniLM, about 90 MB)
.\.venv\Scripts\python.exe scripts/evaluate_rag.py           # 50 products x 4 arms + judge (about 2 h on CPU, resumable; needs Ollama)
.\.venv\Scripts\python.exe scripts/make_handcheck_sheet.py   # blind page to rate 10 listings
```

```python
from catalogiq.llm.client import get_llm_client
from catalogiq.rag.retrieval import HybridRetriever
from catalogiq.rag.listing import ListingGenerator
gen = ListingGenerator(get_llm_client(), HybridRetriever())
result, attributes = gen.generate_from_image("photo.jpg", {"brand": "Puma"}, use_rag=True, self_correct=True)
result.listing, result.report.to_dict()
```

Key finding: retrieved example listings improved style but made a 3B model copy unsupported facts (70% of listings), so the guardrails are
essential. Details: [docs/model_card_listing_generator.md](docs/model_card_listing_generator.md),
[docs/style_guide.md](docs/style_guide.md), [notebooks/04_listing_generator.ipynb](notebooks/04_listing_generator.ipynb).

## Shopper Q&A over reviews (Phase 5)

```powershell
.\.venv\Scripts\python.exe scripts/build_review_index.py     # chunk 139k reviews and index them in ChromaDB (resumable)
.\.venv\Scripts\python.exe scripts/evaluate_qa.py all        # retrieval, answers, judge, summaries (needs Ollama; about 25 min)
.\.venv\Scripts\python.exe scripts/summarise_qa_eval.py      # collect metrics, log to MLflow
```

```python
from catalogiq.rag.qa import ReviewQA
# qa = ReviewQA(get_llm_client(), chunks, collection, titles, min_similarity=0.52)
# qa.answer("B0BYCZ2W78", "Does this shirt run big or small?").output.model_dump()
```

Answers cite review ids, and the system says "Not enough reviews mention this." when fewer than 2 reviews support an answer. On 60 test
questions it refused 14 of 15 unanswerable ones and also refused 31% of answerable ones; cited review ids were never invented. Details and
limits: [docs/model_card_review_qa.md](docs/model_card_review_qa.md), [notebooks/05_review_qa.ipynb](notebooks/05_review_qa.ipynb).

## Catalog quality checker (Phase 6)

```powershell
.\.venv\Scripts\python.exe scripts/tune_quality_thresholds.py   # tune image-check thresholds on the validation split
.\.venv\Scripts\python.exe scripts/evaluate_consistency.py      # inject 300 mismatches, tune on val, report on test
.\.venv\Scripts\python.exe scripts/evaluate_image_checks.py     # 500 clean + 8,500 degraded test photos (about 20 min on CPU)
.\.venv\Scripts\python.exe scripts/run_quality_report.py        # whole test catalog -> reports/phase6/quality_report.csv
```

```python
from catalogiq.quality.image_checks import check_image
from catalogiq.quality.score import listing_score
res = check_image(photo_rgb_uint8)                   # {"metrics", "flags", "issues"}
listing_score(res["issues"], mismatches=[], completeness_pct=100.0, missing=[])   # score 0-100, flags with fixes
```

Every listing gets a 0-100 score and a plain-English fix per flag. On the 6,339 test listings 8.3% carry a flag and 4.1% score below 85; photo-listing
mismatch detection finds about half of 300 injected errors (precision 0.47, a lower bound); jewellery and watches are over-flagged by global thresholds.
Details and limits: [docs/model_card_quality_checker.md](docs/model_card_quality_checker.md), [notebooks/06_quality_checker.ipynb](notebooks/06_quality_checker.ipynb).

## Unified evaluation and scorecard (Phase 7)

```powershell
.\tasks.ps1 eval                                    # re-run every evaluation (about 35 minutes on CPU; needs Ollama for the system test)
.\.venv\Scripts\python.exe -m catalogiq.eval.run_all --render-only   # rebuild the scorecard from saved results
```

`reports/scorecard.md` has one table per component (vision, visual search, listing RAG, review Q&A, quality checker, system), a drift check against the numbers
the earlier phases recorded, a product-impact section whose assumptions are labelled, and the limitations and risks. Results are logged to the MLflow
experiment `catalogiq-eval`, tagged with the git commit. To replace the placeholder time assumptions with your own, time 5 listings with
`reports/phase7/listing_timing.html`, save the CSV as `reports/phase7/manual_listing_times.csv` and use `--render-only`.
Limitations: [docs/limitations_and_risks.md](docs/limitations_and_risks.md).

## API and app (Phase 8)

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000     # API, docs at http://127.0.0.1:8000/docs (models load in about 2 minutes)
.\.venv\Scripts\python.exe -m streamlit run app/Home.py            # app at http://localhost:8501
```

Four pages: **Seller Studio** (photo -> editable attributes -> listing with fact check -> quality score with fix tips -> similar listings), **Shopper View** (search by words or
photo, outfits, review Q&A with citations), **Catalog Ops** (quality KPIs, worst listings, duplicate queue) and **Model Card** (the scorecard). Demo mode (sidebar toggle) uses 10
preloaded products whose listings and sample questions are cached, so they are instant; anything new runs the local model (seconds to a minute on CPU).
API reference: [docs/api.md](docs/api.md). Screenshots: [docs/screenshots/](docs/screenshots/). Checks: `python -m scripts.e2e_smoke` clicks through every flow in headless Edge.

| Seller Studio | Shopper View | Catalog Ops |
|---|---|---|
| ![Seller Studio](docs/screenshots/02_seller_studio_shirt.png) | ![Shopper View](docs/screenshots/04_shopper_search_and_outfit.png) | ![Catalog Ops](docs/screenshots/06_catalog_ops.png) |

## Tests, CI and Docker (Phase 9)

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=src/catalogiq --cov=api     # 222 tests, about 75% coverage, no models or data needed
```

- **Tests and coverage:** [docs/testing.md](docs/testing.md). CI (badge above) runs ruff, black and the tests on every push.
- **Docker:** `docker compose up` runs the API, the app and Ollama; see [docs/docker.md](docs/docker.md) first (disk usage, one-time model download). The files are written but not yet built.
- **Robustness:** friendly errors when Ollama is down or model files are missing, request-size and rate limits ([docs/api.md](docs/api.md)).
- **Performance:** [reports/phase9/performance.md](reports/phase9/performance.md). **Configuration:** [docs/configuration.md](docs/configuration.md).
