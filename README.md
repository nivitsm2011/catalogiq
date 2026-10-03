# CatalogIQ

An AI copilot for e-commerce marketplaces: it helps small sellers create better listings, helps shoppers
get grounded answers, and helps catalog teams monitor quality. Runs fully locally on CPU.

> Status: Phase 1 complete (data pipeline and EDA). Full README coming as features land.

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
