# CatalogIQ

An AI copilot for e-commerce marketplaces: it helps small sellers create better listings, helps shoppers
get grounded answers, and helps catalog teams monitor quality. Runs fully locally on CPU.

> Status: Phase 2 complete (attribute classifier). Full README coming as features land.

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
