# Reproducing CatalogIQ step by step

The commands below were used to build every phase, in this order (Windows PowerShell, from the project root, after
`.\tasks.ps1 setup`). Data and trained models are **not** in git: they are rebuilt locally. Sizes and times are the ones measured
on the development laptop (CPU only). Large downloads are marked; the project budget is about 5 GB of disk.

Order that the API and app need: Phase 1 (data) -> Phase 2 (classifier) -> Phase 3 (CLIP embeddings) -> Phase 4 (knowledge base, for
the listing generator) -> Phase 5 (review index). Phases 6 and 7 produce the quality report and the scorecard (the report and
the scorecard are already committed). The evaluation scripts (`evaluate_rag.py`, `evaluate_qa.py`) also fill the LLM reply cache that
makes the 10 demo products instant; without that cache a demo product runs the local model live (about 15-40 s).

## Data pipeline (Phase 1)

Data is stored locally only and is gitignored. Windows PowerShell:

```powershell
kaggle datasets download paramaggarwal/fashion-product-images-small -p data/raw/fashion --unzip   # 593 MB
.\.venv\Scripts\python.exe scripts/build_fashion.py     # audit, clean, group-aware 70/15/15 splits
.\.venv\Scripts\python.exe scripts/build_reviews.py     # streams Amazon_Fashion, writes ~18 MB sample
```

Kaggle needs a token in `~/.kaggle/access_token`. See [docs/data_card.md](data_card.md) and
[notebooks/01_data_understanding.ipynb](../notebooks/01_data_understanding.ipynb).

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

Test results, error analysis and limitations: [docs/model_card_classifier.md](model_card_classifier.md) and
[notebooks/02_attribute_classifier.ipynb](../notebooks/02_attribute_classifier.ipynb). Trained weights are not in git.

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

Design, outfit rules and evaluation protocol: [docs/search_design.md](search_design.md);
results and failure cases: [notebooks/03_visual_search.ipynb](../notebooks/03_visual_search.ipynb).

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
essential. Details: [docs/model_card_listing_generator.md](model_card_listing_generator.md),
[docs/style_guide.md](style_guide.md), [notebooks/04_listing_generator.ipynb](../notebooks/04_listing_generator.ipynb).

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
limits: [docs/model_card_review_qa.md](model_card_review_qa.md), [notebooks/05_review_qa.ipynb](../notebooks/05_review_qa.ipynb).

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
Details and limits: [docs/model_card_quality_checker.md](model_card_quality_checker.md), [notebooks/06_quality_checker.ipynb](../notebooks/06_quality_checker.ipynb).

## Unified evaluation and scorecard (Phase 7)

```powershell
.\tasks.ps1 eval                                    # re-run every evaluation (about 35 minutes on CPU; needs Ollama for the system test)
.\.venv\Scripts\python.exe -m catalogiq.eval.run_all --render-only   # rebuild the scorecard from saved results
```

`reports/scorecard.md` has one table per component (vision, visual search, listing RAG, review Q&A, quality checker, system), a drift check against the numbers
the earlier phases recorded, a product-impact section whose assumptions are labelled, and the limitations and risks. Results are logged to the MLflow
experiment `catalogiq-eval`, tagged with the git commit. To replace the placeholder time assumptions with your own, time 5 listings with
`reports/phase7/listing_timing.html`, save the CSV as `reports/phase7/manual_listing_times.csv` and use `--render-only`.
Limitations: [docs/limitations_and_risks.md](limitations_and_risks.md).
