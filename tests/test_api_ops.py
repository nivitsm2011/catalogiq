"""API: ops dashboard, duplicate queue, scorecard, catalog photos, warm-up (fake state)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import services
from api.main import create_app
from api.state import AppState
from catalogiq.config import resolve_path
from tests.test_api import make_state

QUALITY_ROWS = [
    # id, masterCategory, articleType, group, score, needs_attention, issues, mismatches, missing
    (1, "Apparel", "Shirts", "topwear", 100.0, False, "", "", "", ""),
    (
        2,
        "Footwear",
        "Casual Shoes",
        "footwear",
        60.0,
        True,
        "blurry;too_dark",
        "baseColour",
        "",
        "",
    ),
    (3, "Apparel", "Shirts", "topwear", 75.0, True, "", "articleType", "usage", "Check baseColour"),
    (4, "Accessories", "Watches", "watches", 90.0, False, "blurry", "", "season;usage", ""),
]


@pytest.fixture()
def quality_csv(tmp_path: Path) -> Path:
    cols = [
        "id",
        "masterCategory",
        "articleType",
        "group",
        "score",
        "needs_attention",
        "image_issues",
        "mismatches",
        "missing",
        "fixes",
    ]
    path = tmp_path / "quality_report.csv"
    pd.DataFrame(QUALITY_ROWS, columns=cols).to_csv(path, index=False)
    return path


def with_duplicate_pair() -> AppState:
    """Items 1 and 2 get the same embedding, so they form one near-duplicate pair."""
    state = make_state()
    state.image_emb = state.image_emb.copy()
    state.image_emb[1] = state.image_emb[0]
    return state


@pytest.fixture()
def ops(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, quality_csv: Path):
    monkeypatch.setattr(services, "_decisions_path", lambda: tmp_path / "decisions.json")
    monkeypatch.setattr(AppState, "quality_paths", staticmethod(lambda: {"batch_csv": quality_csv}))
    monkeypatch.setattr(
        "catalogiq.search.duplicates.load_phash",
        lambda ids: pd.Series("0000000000000000", index=list(ids)),
    )
    monkeypatch.setattr("api.main.llm_reachable", lambda: True)
    with TestClient(create_app(with_duplicate_pair())) as client:
        yield client


# --------------------------------------------------------------------------------- ops summary
def test_ops_summary_counts_issues_by_type_and_category(ops: TestClient) -> None:
    r = ops.get("/ops/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["n_listings"] == 4 and body["avg_score"] == pytest.approx(81.25)
    assert body["pct_with_issue"] == pytest.approx(75.0)  # listings 2, 3 and 4 have an issue
    assert body["pct_needs_attention"] == pytest.approx(50.0)
    counts = body["issue_counts"]
    assert counts["blurry"] == 2 and counts["too_dark"] == 1 and counts["mismatch_baseColour"] == 1
    assert counts["missing_usage"] == 2 and "missing_season" in counts
    assert sum(body["score_hist"]) == 4
    cats = {c["masterCategory"]: c for c in body["by_master_category"]}
    assert cats["Apparel"]["n"] == 2 and cats["Apparel"]["pct_with_issue"] == pytest.approx(50.0)
    groups = [g["group"] for g in body["by_group"]]
    assert groups[0] != "topwear" or len(groups) == 3  # sorted by share of listings with issues


def test_ops_worst_listings_are_sorted_with_friendly_fixes(ops: TestClient) -> None:
    worst = ops.get("/ops/summary", params={"limit": 2}).json()["worst"]
    assert [w["id"] for w in worst] == [2, 3]  # lowest scores first
    assert worst[0]["issues"] == ["blurry", "too_dark"] and worst[0]["quality_score"] == 60.0
    assert worst[1]["fixes"] == ["Check colour"]  # field names are shown as words
    only_shoes = ops.get("/ops/summary", params={"category": "Footwear"}).json()["worst"]
    assert [w["id"] for w in only_shoes] == [2]


def test_ops_summary_without_a_batch_report_is_a_clear_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        AppState, "quality_paths", staticmethod(lambda: {"batch_csv": tmp_path / "nope.csv"})
    )
    with TestClient(create_app(make_state())) as c:
        r = c.get("/ops/summary")
    assert r.status_code == 404 and r.json()["error"]["code"] == "no_quality_report"


# ----------------------------------------------------------------------------- duplicate queue
def test_duplicate_queue_flags_the_identical_pair_and_remembers_decisions(ops: TestClient) -> None:
    q = ops.get("/ops/duplicates").json()
    assert q["n_pairs"] >= 1 and q["n_decided"] == 0
    pair = q["pairs"][0]
    assert {pair["id_a"], pair["id_b"]} == {1, 2} and pair["similarity"] > 0.99
    assert pair["image_a"] == "/catalog/image/1" and pair["decision"] is None
    ops.post("/ops/duplicates/decision", json={"id_a": 1, "id_b": 2, "decision": "keep_both"})
    after = ops.get("/ops/duplicates").json()
    assert after["n_decided"] == 1 and after["pairs"] == []  # decided pairs leave the open queue


# ------------------------------------------------------------------------ scorecard and photos
def test_scorecard_endpoints_read_the_report_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "scorecard.md").write_text("# Scorecard\nall good", encoding="utf-8")
    results = {
        "components": {"vision": {"metrics": {"vision.mean_macro_f1": 0.77, "skipped": None}}},
        "drift": [{"status": "match"}, {"status": "match"}, {"status": "DRIFT"}],
        "git": {"commit": "abc"},
        "generated": "2026-11-13",
    }
    (tmp_path / "eval_results.json").write_text(json.dumps(results), encoding="utf-8")
    mapping = {
        "reports/scorecard.md": tmp_path / "scorecard.md",
        "reports/phase7/eval_results.json": tmp_path / "eval_results.json",
    }
    monkeypatch.setattr("api.main.resolve_path", lambda p: mapping.get(str(p), resolve_path(p)))
    with TestClient(create_app(make_state())) as c:
        assert c.get("/scorecard").text.startswith("# Scorecard")
        m = c.get("/scorecard/metrics").json()
    assert m["metrics"] == {"vision.mean_macro_f1": 0.77} and m["git"] == {"commit": "abc"}
    assert m["drift"] == {"match": 2, "DRIFT": 1}


def test_missing_scorecard_says_how_to_create_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("api.main.resolve_path", lambda p: tmp_path / "absent")
    with TestClient(create_app(make_state())) as c:
        for path in ("/scorecard", "/scorecard/metrics"):
            r = c.get(path)
            assert r.status_code == 404 and "run_all" in r.json()["error"]["message"]


def test_catalog_photo_is_served_with_a_cache_header(
    monkeypatch: pytest.MonkeyPatch, photo_dir: Path
) -> None:
    monkeypatch.setattr(
        "catalogiq.data.fashion.image_path", lambda _dir, _id: photo_dir / "shirt_blue.jpg"
    )
    with TestClient(create_app(make_state())) as c:
        r = c.get("/catalog/image/1")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert "max-age" in r.headers["cache-control"] and r.content[:2] == b"\xff\xd8"


# --------------------------------------------------------------------------------- warm-up
def test_warm_up_runs_every_step_and_survives_failures(
    monkeypatch: pytest.MonkeyPatch, quality_csv: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(services, "_decisions_path", lambda: tmp_path / "d.json")
    monkeypatch.setattr(AppState, "quality_paths", staticmethod(lambda: {"batch_csv": quality_csv}))
    monkeypatch.setattr(
        "catalogiq.search.duplicates.load_phash",
        lambda ids: pd.Series("0000000000000000", index=list(ids)),
    )
    calls: list[Any] = []

    class Retriever:
        def search(self, query: str, k: int, group: Any) -> None:
            calls.append(query)
            raise RuntimeError("chroma is not available")  # must not stop the other steps

    state = with_duplicate_pair()
    state.generator = type("G", (), {"retriever": Retriever()})()
    services.warm_up(state)
    assert calls == ["shirt"]
    assert {"review_text", "quality_report", "duplicate_pairs"} <= set(state._lazy)


def test_lazy_builds_once_and_review_text_is_indexed() -> None:
    state = make_state()
    built: list[int] = []

    def build() -> int:
        built.append(1)
        return 42

    assert state.lazy("x", build) == 42 and state.lazy("x", build) == 42 and len(built) == 1
    assert list(state.review_text.index) == ["R1", "R2"]
    assert np.isfinite(state.image_emb).all()
