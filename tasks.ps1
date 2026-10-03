# Task runner for Windows (no `make` available). Usage: .\tasks.ps1 <setup|lint|test|app|api>
param([Parameter(Position = 0)][string]$Task = "help")

$py = ".\.venv\Scripts\python.exe"

switch ($Task) {
    "setup" {
        if (-not (Test-Path $py)) { uv venv .venv --python cpython-3.11-windows-x86_64-none }
        uv pip install --python $py -r requirements.txt
        uv pip install --python $py -e .
    }
    "lint" {
        & $py -m ruff check .
        & $py -m black --check .
    }
    "test" { & $py -m pytest }
    "app" { & $py -m streamlit run app/main.py }
    "api" { & $py -m uvicorn api.main:app --reload }
    default { Write-Host "Usage: .\tasks.ps1 <setup|lint|test|app|api>" }
}
