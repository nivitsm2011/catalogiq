"""Tests for the Phase 7 evaluation harness (no models, no data files needed)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from catalogiq.eval.common import ComponentResult, CountingClient, git_info, table_to_markdown
from catalogiq.eval.impact import ASSUMPTION, ILLUSTRATIVE, MEASURED, build_impact
from catalogiq.eval.reference import drift_table, load_reference, tolerance
from catalogiq.eval.run_all import _finite, _flat_metrics, _mlflow_name, result_to_dict
from catalogiq.eval.scorecard import _drift, _pct, _safe
from catalogiq.llm.client import LLMClient, LLMResponse


class _Echo(LLMClient):
    model = "fake"

    def chat(self, *args, **kwargs) -> LLMResponse:  # type: ignore[override]
        return LLMResponse("ok", 0.0)


# ----------------------------------------------------------------------------------- drift
def test_drift_statuses() -> None:
    ref = {
        "a": {"value": 0.806, "tol": "abs", "source": "t"},
        "b": {"value": 0.70, "tol": "abs2", "source": "t"},
        "c": {"value": 18.7, "tol": "timing", "source": "t"},
        "d": {"value": 0.5, "tol": "abs", "source": "t"},
        "e": {"value": 8.3, "tol": 0.06, "source": "t"},
        "f": {"value": 1.0, "tol": "abs", "source": "t"},
    }
    now = {"a": 0.8062, "b": 0.74, "c": 11.0, "d": 0.51, "e": 8.31}
    t = drift_table(now, ref).set_index("metric")
    assert t.loc["a", "status"] == "match"
    assert t.loc["b", "status"] == "DRIFT"  # 0.04 > 0.006
    assert t.loc["c", "status"] == "timing"  # within 50% of 18.7
    assert t.loc["d", "status"] == "DRIFT"
    assert t.loc["e", "status"] == "match"  # numeric tolerance
    assert t.loc["f", "status"] == "not re-run"
    assert math.isclose(t.loc["a", "diff"], 0.0002, abs_tol=1e-9)


def test_timing_outside_band_is_drift() -> None:
    spec = {"value": 10.0, "tol": "timing", "source": "t"}
    tol, kind = tolerance(spec, 30.0)
    assert kind == "timing" and tol == pytest.approx(5.0)
    assert drift_table({"x": 30.0}, {"x": spec}).loc[0, "status"] == "DRIFT"


def test_reference_file_is_well_formed() -> None:
    ref = load_reference()
    assert len(ref) > 60
    for key, spec in ref.items():
        assert {"value", "tol", "source"} <= set(spec), key
        assert key.split(".")[0] in {
            "vision",
            "search",
            "duplicates",
            "listing",
            "qa",
            "quality",
            "consistency",
        }


# ---------------------------------------------------------------------------------- impact
def test_impact_without_timings_is_illustrative() -> None:
    res = build_impact(
        {"quality.batch.pct_with_any_issue": 8.3, "quality.batch.pct_needs_attention": 4.1},
        None,
        None,
    )
    inputs = {i["input"]: i for i in res["inputs"]}
    assert all(i["status"] == ASSUMPTION for i in inputs.values())
    money = [
        r
        for r in res["rows"]
        if "cost" in r["quantity"].lower() or "hours" in r["quantity"].lower()
    ]
    assert money and all(r["status"] == ILLUSTRATIVE for r in money)
    assert res["timings_supplied"] is False and len(res["not_estimated"]) >= 3


def test_impact_with_owner_timings() -> None:
    timings = pd.DataFrame(
        {"product_id": [1, 2], "manual_minutes": [10.0, 14.0], "review_minutes": [2.0, 4.0]}
    )
    res = build_impact({}, None, timings)
    inputs = {i["input"]: i for i in res["inputs"]}
    assert inputs["Time to write one listing by hand"]["status"] == MEASURED
    assert inputs["Time to write one listing by hand"]["value"] == 12.0
    assert inputs["Time to review and fix one generated draft"]["value"] == 3.0
    saved = next(r for r in res["rows"] if r["quantity"].startswith("Hands-on minutes saved"))
    assert saved["status"] == MEASURED and "9.0 min" in saved["value"]
    monthly = next(r for r in res["rows"] if r["quantity"].startswith("Hands-on hours"))
    assert monthly["status"] == ILLUSTRATIVE  # volume is still an assumption


def test_manual_timings_alone_do_not_make_the_saving_measured() -> None:
    timings = pd.DataFrame({"product_id": [1], "manual_minutes": [10.0]})
    res = build_impact({}, None, timings)
    saved = next(r for r in res["rows"] if r["quantity"].startswith("Hands-on minutes saved"))
    assert saved["status"] == ILLUSTRATIVE  # the review time is still a placeholder


# ---------------------------------------------------------------------------------- helpers
def test_counting_client_enforces_the_budget() -> None:
    client = CountingClient(_Echo(), max_calls=2)
    client.chat([{"role": "user", "content": "x"}])
    client.chat([{"role": "user", "content": "x"}])
    assert client.calls == 2
    with pytest.raises(RuntimeError, match="budget"):
        client.chat([{"role": "user", "content": "x"}])
    unlimited = CountingClient(_Echo())
    for _ in range(5):
        unlimited.chat([])
    assert unlimited.calls == 5


def test_git_info_has_a_commit_hash() -> None:
    info = git_info()
    assert len(info["commit"]) == 40 and info["short"] == info["commit"][:7]
    assert isinstance(info["dirty"], bool)


def test_table_to_markdown_and_names() -> None:
    md = table_to_markdown(pd.DataFrame({"a": [1.23456, float("nan")], "b": ["x", "y"]}))
    assert md.splitlines()[0] == "| a | b |" and "1.235" in md
    assert _mlflow_name("search.clip.recall@10") == "search.clip.recall_at_10"
    assert _finite({"a": 1.0, "b": float("nan"), "c": None, "d@": 2}) == {"a": 1.0, "d_at_": 2.0}
    assert _pct(0.1234) == "12.3%" and _pct(float("nan")) == ""


def test_result_serialisation_and_flattening() -> None:
    res = ComponentResult(
        "x", {"m": 1.0, "n": float("nan")}, {"t": pd.DataFrame({"a": [1]})}, {"p": 1}, 2.0
    )
    d = result_to_dict(res)
    assert d["metrics"]["n"] is None and d["tables"]["t"] == [{"a": 1}]
    flat = _flat_metrics({"components": {"x": d}})
    assert flat == {"m": 1.0}


def test_scorecard_helpers_degrade_gracefully() -> None:
    assert "Not available" in _safe(lambda: {}["missing"])
    drift = pd.DataFrame(
        [
            {
                "metric": "vision.x",
                "previous": 1.0,
                "now": 1.0,
                "diff": 0.0,
                "status": "match",
                "source": "s",
            }
        ]
    )
    assert "Every number matched" in _drift(drift, [])
    drift.loc[0, ["now", "diff", "status"]] = [2.0, 1.0, "DRIFT"]
    text = _drift(drift, [{"match": "vision", "note": "because of the test"}])
    assert "because of the test" in text and "1 DRIFT" in text
