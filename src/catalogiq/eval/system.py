"""System measurements: photo -> full listing latency on CPU, peak RAM and disk footprint.

The end-to-end run uses an UNCACHED LLM so the model really runs. RAM is sampled from this process
and from every Ollama process (the model server holds the language model's weights).
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from PIL import Image

from catalogiq.config import PROJECT_ROOT, get_config, get_eval_config, resolve_path
from catalogiq.logging_utils import get_logger

logger = get_logger(__name__)
GB = 1024**3


class RamSampler:
    """Background thread recording the peak resident memory of this process and of Ollama."""

    def __init__(self, interval_s: float) -> None:
        self.interval = interval_s
        self.peak_python = 0
        self.peak_ollama = 0
        self.peak_combined = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._me = psutil.Process(os.getpid())

    def _ollama_rss(self) -> int:
        total = 0
        for proc in psutil.process_iter(["name", "memory_info"]):
            name = (proc.info["name"] or "").lower()
            if ("ollama" in name or "llama-server" in name) and proc.info["memory_info"]:
                total += proc.info["memory_info"].rss
        return total

    def _run(self) -> None:
        while not self._stop.is_set():
            py, ol = self._me.memory_info().rss, self._ollama_rss()
            self.peak_python = max(self.peak_python, py)
            self.peak_ollama = max(self.peak_ollama, ol)
            self.peak_combined = max(self.peak_combined, py + ol)
            time.sleep(self.interval)

    def __enter__(self) -> RamSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda e: None):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def disk_footprint() -> dict[str, Any]:
    """Size of the project folders and of the Ollama models stored outside the project."""
    cfg = get_eval_config()["system"]
    parts = {}
    for name in (".venv", "data", "models", "mlruns", "reports", "docs", "notebooks", ".git"):
        p = PROJECT_ROOT / name
        parts[name] = _dir_size(p) / GB if p.exists() else 0.0
    total_project = _dir_size(PROJECT_ROOT) / GB
    parts["other (src, tests, scripts, configs, ...)"] = max(
        0.0, total_project - sum(parts.values())
    )
    sub = {}
    for top in ("data", "models"):
        for child in (
            sorted((PROJECT_ROOT / top).iterdir()) if (PROJECT_ROOT / top).exists() else []
        ):
            size = (_dir_size(child) if child.is_dir() else child.stat().st_size) / GB
            if size >= 0.05:
                sub[f"{top}/{child.name}"] = size
    ollama = Path(cfg["ollama_models_dir"]).expanduser()
    ollama_gb = _dir_size(ollama) / GB if ollama.exists() else float("nan")
    used, others = ollama_models()
    return {
        "project_total_gb": total_project,
        "parts_gb": parts,
        "large_items_gb": dict(sorted(sub.items(), key=lambda kv: -kv[1])),
        "ollama_models_gb": ollama_gb,
        "ollama_used_gb": sum(m["gb"] for m in used),
        "ollama_used": used,
        "ollama_other": others,
        "budget_gb": cfg["budget_gb"],
        "total_with_ollama_gb": total_project + sum(m["gb"] for m in used),
    }


def ollama_models() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Models in the Ollama store: (used by CatalogIQ, other models that are not part of it)."""
    wanted = get_config()["models"]["llm"]["model"]
    out = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True, check=False, timeout=60
    )
    used, others = [], []
    for line in out.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        gb = float(parts[2]) / (1.0 if parts[3].upper() == "GB" else 1024.0)
        (used if parts[0] == wanted else others).append({"name": parts[0], "gb": gb})
    return used, others


def _ensure_ollama() -> None:
    """Listing models restarts the Ollama server if it has stopped (a known laptop quirk)."""
    subprocess.run(["ollama", "list"], capture_output=True, check=False, timeout=60)


def end_to_end(n_photos: int | None = None) -> dict[str, Any]:
    """Photo -> quality check -> attributes -> full listing, timed per stage, uncached LLM.

    Returns:
        ``rows`` (one per photo and mode), ``summary`` (mean / median / p95 seconds per stage and
        mode), ``ram`` (peaks in MB), ``cold_start_s`` (first LLM call) and the machine description.
    """
    from catalogiq.data.fashion import image_path
    from catalogiq.llm.client import get_llm_client
    from catalogiq.quality.clip_checks import prompt_embeddings, text_margin
    from catalogiq.quality.image_checks import check_image
    from catalogiq.rag.evaluate import load_brand_lexicon, sample_products, seller_notes_for
    from catalogiq.rag.listing import ListingGenerator
    from catalogiq.rag.retrieval import HybridRetriever
    from catalogiq.search.embeddings import ClipEncoder
    from catalogiq.vision.predict import AttributePredictor

    cfg = get_eval_config()["system"]
    n = n_photos or cfg["n_photos"]
    _ensure_ollama()
    products = sample_products().head(n)
    images_dir = resolve_path(get_config()["data"]["fashion"]["images_dir"])
    rows: list[dict[str, Any]] = []
    with RamSampler(cfg["sample_interval_s"]) as ram:
        t0 = time.perf_counter()
        predictor, encoder = AttributePredictor(), ClipEncoder()
        prompts = prompt_embeddings(encoder)
        gen = ListingGenerator(get_llm_client(cache=False), HybridRetriever(), load_brand_lexicon())
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        gen.llm.chat(
            [{"role": "user", "content": "Reply with the single word: ready"}], temperature=0.0
        )
        cold_start_s = time.perf_counter() - t0
        for prod in products.to_dict("records"):
            path = image_path(images_dir, int(prod["id"]))
            t = time.perf_counter()
            photo = np.asarray(Image.open(path).convert("RGB"))
            check = check_image(
                photo, text_margin=float(text_margin(encoder.encode_image(photo), prompts)[0])
            )
            quality_s = time.perf_counter() - t
            t = time.perf_counter()
            attributes = predictor.predict(photo)
            classify_s = time.perf_counter() - t
            for mode, (use_rag, fix) in cfg["modes"].items():
                t = time.perf_counter()
                result = gen.generate(attributes, seller_notes_for(pd.Series(prod)), use_rag, fix)
                gen_s = time.perf_counter() - t
                rows.append(
                    {
                        "product_id": int(prod["id"]),
                        "mode": mode,
                        "quality_s": quality_s,
                        "classify_s": classify_s,
                        "generate_s": gen_s,
                        "total_s": quality_s + classify_s + gen_s,
                        "ok": result.listing is not None,
                        "issues_found": len(check["issues"]),
                    }
                )
                logger.info("photo %s %s: %.1fs", prod["id"], mode, rows[-1]["total_s"])
    df = pd.DataFrame(rows)
    summary = {}
    for mode, g in df.groupby("mode"):
        summary[mode] = {
            stage: {
                "mean": float(g[stage].mean()),
                "median": float(g[stage].median()),
                "p95": float(np.percentile(g[stage], 95)),
            }
            for stage in ("quality_s", "classify_s", "generate_s", "total_s")
        }
    return {
        "rows": df.to_dict("records"),
        "summary": summary,
        "n_photos": int(len(products)),
        "load_models_s": load_s,
        "cold_start_s": cold_start_s,
        "ram": {
            "python_mb": ram.peak_python / 1e6,
            "ollama_mb": ram.peak_ollama / 1e6,
            "combined_mb": ram.peak_combined / 1e6,
            "machine_total_mb": psutil.virtual_memory().total / 1e6,
        },
        "machine": {
            "cpu_threads": os.cpu_count(),
            "note": "CPU only, x64 Python emulated on ARM64",
        },
    }
