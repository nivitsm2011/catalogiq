"""Product impact in business terms, with every input labelled MEASURED or ASSUMPTION.

Rules (from the project brief): assumptions are never presented as results; a number that depends on
an assumption is labelled ILLUSTRATIVE; what the data cannot support is listed as such.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from catalogiq.config import get_eval_config, resolve_path

MEASURED, ASSUMPTION, ILLUSTRATIVE = (
    "MEASURED",
    "ASSUMPTION",
    "ILLUSTRATIVE (depends on assumptions)",
)


def load_timings() -> pd.DataFrame | None:
    """The owner's timings (``product_id, manual_minutes[, review_minutes]``) or None."""
    path = resolve_path(get_eval_config()["paths"]["timings"])
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if "manual_minutes" in df and df["manual_minutes"].notna().any() else None


def build_impact(
    metrics: dict[str, float], system: dict[str, Any] | None, timings: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Convert measured metrics and labelled assumptions into business quantities.

    Returns:
        ``inputs`` (name, value, status, basis), ``rows`` (quantity, value, status, how it is
        computed) and ``not_estimated`` (things the data cannot support).
    """
    a = dict(get_eval_config()["impact"]["assumptions"])
    inputs: list[dict[str, Any]] = []

    def take(name: str, label: str, unit: str) -> float:
        value, status, basis = float(a[name]), ASSUMPTION, "configs/eval.yaml (placeholder)"
        if name == "manual_minutes_per_listing" and timings is not None:
            value, status = float(timings["manual_minutes"].mean()), MEASURED
            basis = (
                f"owner timed {int(timings['manual_minutes'].notna().sum())} manual listings (mean)"
            )
        if (
            name == "review_minutes_per_generated_listing"
            and timings is not None
            and "review_minutes" in timings
            and timings["review_minutes"].notna().any()
        ):
            value, status = float(timings["review_minutes"].mean()), MEASURED
            basis = (
                f"owner timed {int(timings['review_minutes'].notna().sum())} draft reviews (mean)"
            )
        inputs.append(
            {"input": label, "value": value, "unit": unit, "status": status, "basis": basis}
        )
        return value

    manual = take("manual_minutes_per_listing", "Time to write one listing by hand", "min")
    review = take(
        "review_minutes_per_generated_listing", "Time to review and fix one generated draft", "min"
    )
    ops_unscreened = take(
        "ops_review_minutes_per_listing_unscreened", "Ops spot check of one listing", "min"
    )
    ops_flagged = take(
        "ops_review_minutes_per_flagged_listing", "Ops review of one flagged listing", "min"
    )
    volume = take("listings_per_month", "Listings per month (illustrative marketplace)", "listings")
    hourly = take("hourly_cost", "Labour cost per hour (illustrative)", "currency units")

    rows: list[dict[str, Any]] = []
    status_inputs = {i["input"]: i["status"] for i in inputs}
    manual_status = status_inputs["Time to write one listing by hand"]
    review_status = status_inputs["Time to review and fix one generated draft"]

    def derived(statuses: list[str]) -> str:
        return MEASURED if all(s == MEASURED for s in statuses) else ILLUSTRATIVE

    if system:
        slow = system["summary"].get("rag_with_rewrite", {}).get("total_s", {}).get("mean")
        fast = system["summary"].get("no_rag", {}).get("total_s", {}).get("mean")
        if fast is not None and slow is not None:
            rows.append(
                {
                    "quantity": "Machine time from photo to draft listing (CPU)",
                    "value": f"{fast:.0f} s (no-RAG) to {slow:.0f} s (RAG with rewrite)",
                    "status": MEASURED,
                    "how": "end-to-end run, uncached LLM (system section)",
                }
            )
    saved = manual - review
    rows.append(
        {
            "quantity": "Hands-on minutes saved per listing",
            "value": f"{saved:.1f} min ({manual:.1f} manual - {review:.1f} review)",
            "status": derived([manual_status, review_status]),
            "how": "manual time minus review time; machine time is waiting, not hands-on, time",
        }
    )
    hours = saved * volume / 60.0
    rows.append(
        {
            "quantity": "Hands-on hours saved per month",
            "value": f"{hours:,.0f} h at {volume:,.0f} listings",
            "status": ILLUSTRATIVE,
            "how": "minutes saved x listings per month (volume is an assumption)",
        }
    )
    rows.append(
        {
            "quantity": "Labour cost saved per month",
            "value": f"{hours * hourly:,.0f} currency units at {hourly:.0f} per hour",
            "status": ILLUSTRATIVE,
            "how": "hours saved x hourly cost (both assumptions)",
        }
    )
    rate_any = metrics.get("quality.batch.pct_with_any_issue")
    rate_att = metrics.get("quality.batch.pct_needs_attention")
    if rate_any is not None and rate_att is not None:
        rows.append(
            {
                "quantity": "Share of listings the quality checker flags / sends to review",
                "value": f"{rate_any:.1f}% any flag, {rate_att:.1f}% score below 85",
                "status": MEASURED,
                "how": "batch report, 6,339 test listings (includes false alarms; misses unknown)",
            }
        )
        screened = volume * rate_any / 100 * ops_flagged / 60
        unscreened = volume * ops_unscreened / 60
        rows.append(
            {
                "quantity": "Ops review hours per month: look at flagged only vs spot check all",
                "value": f"{screened:,.0f} h vs {unscreened:,.0f} h",
                "status": ILLUSTRATIVE,
                "how": "volume x rate x minutes per review (volume and minutes are assumptions)",
            }
        )
    for arm, label in (
        ("rag", "RAG drafts with an unsupported fact"),
        ("no_rag_fix", "no-RAG + rewrite drafts with an unsupported fact"),
    ):
        v = metrics.get(f"listing.{arm}.hallucination_rate")
        if v is not None:
            rows.append(
                {
                    "quantity": label,
                    "value": f"{100 * v:.0f}%",
                    "status": MEASURED,
                    "how": "Phase 4 guardrails, 50 products; a person must still review drafts",
                }
            )
    ref = metrics.get("qa.correct_refusal_rate")
    answered = metrics.get("qa.answered_answerable")
    if ref is not None and answered is not None:
        rows.append(
            {
                "quantity": "Shopper questions: answered with citations / refused when no evidence",
                "value": f"{100 * answered / 45:.0f}% of answerable answered, "
                f"{100 * ref:.0f}% of unanswerable refused",
                "status": MEASURED,
                "how": "Phase 5, 60 questions",
            }
        )
    not_estimated = [
        "Change in returns: no returns data exists in this project.",
        "Change in conversion or search revenue: needs an A/B test on a real marketplace.",
        "Shopper time saved by Q&A: needs real question traffic, not the 60 test questions.",
        "Seller satisfaction: no seller was asked.",
    ]
    return {
        "inputs": inputs,
        "rows": rows,
        "not_estimated": not_estimated,
        "timings_supplied": timings is not None,
    }
