"""Evaluation harness (run_all, scorecard, system helpers), config, utils and tracking.

Nothing here runs a model: components are replaced by fakes and every output goes to a tmp folder.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import pytest

from catalogiq import config, tracking, utils
from catalogiq.config import PROJECT_ROOT, get_config, get_eval_config
from catalogiq.eval import common, run_all, system
from catalogiq.eval.common import ComponentResult
from catalogiq.eval.scorecard import render

SAVED_RESULTS = PROJECT_ROOT / "reports" / "phase7" / "eval_results.json"


@pytest.fixture()
def eval_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    """Point every output of the evaluation at a tmp folder (the config dict is shared)."""
    paths = {
        "out_dir": tmp_path / "phase7",
        "scorecard": tmp_path / "scorecard.md",
        "results": tmp_path / "phase7" / "eval_results.json",
        "drift": tmp_path / "phase7" / "drift.csv",
        "timings": tmp_path / "phase7" / "manual_listing_times.csv",
        "logs_dir": tmp_path / "phase7" / "logs",
    }
    (tmp_path / "phase7").mkdir()
    for key, value in paths.items():
        monkeypatch.setitem(get_eval_config()["paths"], key, str(value))
    monkeypatch.setitem(get_config()["paths"], "mlflow_tracking_dir", str(tmp_path / "mlruns"))
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return paths


# ------------------------------------------------------------------------------- scorecard
@pytest.mark.skipif(not SAVED_RESULTS.exists(), reason="saved evaluation results not present")
def test_scorecard_renders_every_section_from_the_saved_results() -> None:
    results = json.loads(SAVED_RESULTS.read_text(encoding="utf-8"))
    text = render(results, get_eval_config().get("drift_notes", []))
    for heading in (
        "## 1. Vision",
        "## 2. Visual search",
        "## 3. Listing generator",
        "## 4. Review Q&A",
        "## 5. Catalog quality checker",
        "## 6. System",
        "## 7. Drift check",
        "## 8. Product impact",
        "## 9. Limitations and risks",
        "## 10. Provenance",
    ):
        assert heading in text
    assert "_Not available" not in text  # every component had saved results
    assert "ILLUSTRATIVE" in text or "MEASURED" in text  # assumptions are always labelled


def test_scorecard_marks_missing_components_instead_of_failing() -> None:
    results = {
        "git": {"short": "abc1234", "dirty": False},
        "components": {},
        "drift": [],
        "generated": "2026-01-01 00:00",
    }
    text = render(results)
    assert text.count("_Not available") >= 6 and "Generated 2026-01-01" in text


# --------------------------------------------------------------------------------- run_all
def fake_runner(name: str, calls: list[str]):
    def run(**kwargs: Any) -> ComponentResult:
        calls.append(name)
        return ComponentResult(
            name,
            {f"{name}.score": 0.5, f"{name}.nan": float("nan")},
            {"table": [{"a": 1}]},
            {"recomputed": "yes"},
            1.5,
        )

    return run


def test_run_all_runs_components_logs_to_mlflow_and_writes_the_scorecard(
    monkeypatch: pytest.MonkeyPatch, eval_paths: dict[str, Path]
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        run_all, "RUNNERS", {n: fake_runner(n, calls) for n in ("vision", "qa", "quality")}
    )
    monkeypatch.setattr(run_all, "ORDER", ["vision", "qa", "quality"])
    monkeypatch.setattr(run_all, "disk_footprint", lambda: {"project_total_gb": 1.0})
    results = run_all.run_all(skip_system=True, allow_llm_calls=3)
    assert calls == ["vision", "qa", "quality"]
    assert common.LLM_CALL_BUDGET["max_calls"] == 3
    saved = json.loads(eval_paths["results"].read_text(encoding="utf-8"))
    assert set(saved["components"]) == {"vision", "qa", "quality"}
    assert saved["components"]["vision"]["metrics"]["vision.nan"] is None  # NaN is not valid JSON
    assert eval_paths["scorecard"].read_text(encoding="utf-8").startswith("# CatalogIQ scorecard")
    assert eval_paths["drift"].exists() and results["total_seconds"] >= 0
    # one parent run tagged with the commit and one nested run per component
    exp = mlflow.get_experiment_by_name(get_eval_config()["experiment"])
    runs = mlflow.search_runs([exp.experiment_id], output_format="list")
    assert len(runs) == 4
    assert any(r.data.tags.get("git_commit") for r in runs)


def test_run_all_resume_skips_finished_components(
    monkeypatch: pytest.MonkeyPatch, eval_paths: dict[str, Path]
) -> None:
    done = {"components": {"vision": run_all.result_to_dict(fake_runner("vision", [])())}}
    eval_paths["results"].write_text(json.dumps(done), encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        run_all, "RUNNERS", {n: fake_runner(n, calls) for n in ("vision", "qa", "quality")}
    )
    monkeypatch.setattr(run_all, "ORDER", ["vision", "qa", "quality"])
    monkeypatch.setattr(run_all, "disk_footprint", lambda: {"project_total_gb": 1.0})
    run_all.run_all(only=["vision", "qa"], skip_system=True, resume=True)
    assert calls == ["qa"]  # vision was already saved, quality was not requested


def test_render_only_rebuilds_from_saved_results(
    eval_paths: dict[str, Path],
) -> None:
    if not SAVED_RESULTS.exists():
        pytest.skip("saved evaluation results not present")
    shutil.copy(SAVED_RESULTS, eval_paths["results"])
    run_all.run_all(render_only=True)
    text = eval_paths["scorecard"].read_text(encoding="utf-8")
    assert "## 8. Product impact" in text and eval_paths["drift"].exists()


def test_main_passes_the_command_line_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(run_all, "run_all", lambda **kw: seen.update(kw))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_all", "--only", "vision,qa", "--skip-system", "--resume", "--allow-llm-calls", "7"],
    )
    run_all.main()
    assert seen["only"] == ["vision", "qa"] and seen["skip_system"] and seen["resume"]
    assert seen["allow_llm_calls"] == 7 and seen["render_only"] is False


def test_jsonable_handles_frames_nan_and_nesting() -> None:
    import pandas as pd

    frame = pd.DataFrame({"a": [1, 2]}, index=pd.Index(["x", "y"], name="k"))
    out = run_all._jsonable({"f": frame, "n": float("nan"), "t": (1, {"z": float("nan")})})
    assert out["f"] == [{"k": "x", "a": 1}, {"k": "y", "a": 2}]
    assert out["n"] is None and out["t"] == [1, {"z": None}]
    assert run_all._mlflow_name("recall@10") == "recall_at_10"
    assert run_all._finite({"a@1": 1, "b": float("inf"), "c": "text"}) == {"a_at_1": 1.0}


# ----------------------------------------------------------------------------- common helpers
def test_run_script_returns_seconds_and_reports_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, eval_paths: dict[str, Path]
) -> None:
    scripts = tmp_path / "root" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "ok.py").write_text("print('fine')\n", encoding="utf-8")
    (scripts / "bad.py").write_text("raise SystemExit('boom')\n", encoding="utf-8")
    monkeypatch.setattr(common, "PROJECT_ROOT", tmp_path / "root")
    assert common.run_script("ok") >= 0
    with pytest.raises(RuntimeError, match="boom"):
        common.run_script("bad")
    module = common.load_script("ok")
    assert module.__name__ == "catalogiq_script_ok"


# --------------------------------------------------------------------------------- system helpers
def test_ram_sampler_records_a_positive_peak() -> None:
    with system.RamSampler(0.01) as ram:
        blob = bytearray(20_000_000)  # make the process visibly bigger
        import time

        time.sleep(0.1)
    assert ram.peak_python > 0 and ram.peak_combined >= ram.peak_python and len(blob) > 0


def test_ollama_models_parses_the_list_output(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = (
        "NAME ID SIZE MODIFIED\n"
        "llama3.2:3b abc 2.0 GB 2 days ago\n"
        "nomic-embed-text:latest def 274 MB 3 days ago\n"
        "short line\n"
    )
    monkeypatch.setattr(
        system.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=listing, returncode=0)
    )
    used, others = system.ollama_models()
    assert used == [{"name": "llama3.2:3b", "gb": 2.0}]
    assert others[0]["name"] == "nomic-embed-text:latest" and others[0]["gb"] == pytest.approx(
        274 / 1024
    )


def test_disk_footprint_adds_up_the_project_folders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "proj"
    (root / "data" / "processed").mkdir(parents=True)
    (root / "data" / "processed" / "big.bin").write_bytes(b"0" * 200_000)
    (root / "models").mkdir()
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("x = 1")
    monkeypatch.setattr(system, "PROJECT_ROOT", root)
    monkeypatch.setattr(system, "ollama_models", lambda: ([{"name": "m", "gb": 1.5}], []))
    out = system.disk_footprint()
    assert out["project_total_gb"] > 0 and out["parts_gb"]["data"] > 0
    assert out["ollama_used_gb"] == 1.5
    assert out["total_with_ollama_gb"] == pytest.approx(out["project_total_gb"] + 1.5)
    assert system._dir_size(root / "does-not-exist") == 0


# -------------------------------------------------------------------------------- config / utils
def test_config_helpers_and_environment_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("a: {b: 1}\n", encoding="utf-8")
    assert config.load_config(f) == {"a": {"b": 1}}
    assert config.resolve_path(f) == f and config.resolve_path("x/y") == PROJECT_ROOT / "x" / "y"
    for getter in (
        config.get_vision_config,
        config.get_search_config,
        config.get_rag_config,
        config.get_qa_config,
        config.get_quality_config,
        config.get_eval_config,
        config.get_api_config,
    ):
        assert isinstance(getter(), dict) and getter()
    monkeypatch.setenv(config.LLM_HOST_ENV, "http://ollama:11434")
    config.get_config.cache_clear()
    try:
        assert config.get_config()["models"]["llm"]["host"] == "http://ollama:11434"
    finally:
        monkeypatch.delenv(config.LLM_HOST_ENV)
        config.get_config.cache_clear()
    assert config.get_config()["models"]["llm"]["host"].startswith("http://localhost")


def test_utils_seed_device_timer_and_cache_dirs(monkeypatch: pytest.MonkeyPatch) -> None:
    import random

    import numpy as np
    import torch

    utils.set_seed(5)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    utils.set_seed(5)
    assert a == (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert utils.get_device() in {"cpu", "cuda"}

    @utils.timed
    def add(x: int, y: int) -> int:
        return x + y

    assert add(1, 2) == 3 and add.__name__ == "add"
    monkeypatch.delenv("TORCH_HOME", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    utils.configure_cache_dirs()
    import os

    assert os.environ["TORCH_HOME"].replace("\\", "/").endswith("models/cache/torch")
    assert os.environ["HF_HOME"].replace("\\", "/").endswith("models/cache/huggingface")


def test_tracking_start_run_logs_params_to_a_local_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(get_config()["paths"], "mlflow_tracking_dir", str(tmp_path / "mlruns"))
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    with tracking.start_run("unit-test", {"seed": 42, "long": "x" * 400}) as run:
        mlflow.log_metric("score", 0.9)
        run_id = run.info.run_id
    stored = mlflow.get_run(run_id)
    assert stored.data.params["seed"] == "42" and len(stored.data.params["long"]) == 250
    assert stored.data.metrics["score"] == 0.9 and (tmp_path / "mlruns").exists()
