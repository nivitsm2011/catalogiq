"""Verify that every required package imports and report versions and hardware."""

from __future__ import annotations

import importlib
import platform
import sys

MODULES = [
    "pandas",
    "numpy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "plotly",
    "PIL",
    "cv2",
    "torch",
    "torchvision",
    "open_clip",
    "faiss",
    "sentence_transformers",
    "chromadb",
    "datasets",
    "ollama",
    "mlflow",
    "fastapi",
    "uvicorn",
    "pydantic",
    "streamlit",
    "yaml",
    "dotenv",
    "tqdm",
    "pytest",
    "ruff",
    "black",
    "jupyter",
    "ipykernel",
    "catalogiq",
]


def main() -> int:
    """Import each module, print a table, and return a non-zero exit code on failure."""
    print(f"Python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
    failed: list[str] = []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            print(f"  OK   {name:<22} {getattr(mod, '__version__', '')}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL {name:<22} {type(exc).__name__}: {exc}")
    import torch

    print(f"GPU available: {torch.cuda.is_available()} (CPU fallback is always used otherwise)")
    print("ALL IMPORTS OK" if not failed else f"FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
