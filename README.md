# CatalogIQ

An AI copilot for e-commerce marketplaces: it helps small sellers create better listings, helps shoppers
get grounded answers, and helps catalog teams monitor quality. Runs fully locally on CPU.

> Status: Phase 4 complete (grounded listing generator). Full README coming as features land.

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
