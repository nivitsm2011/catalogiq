"""Time the seller's photo -> listing path against a running API, stage by stage.

Usage (API running on :8000):
    python -m scripts.profile_listing --label before --photos 3 --repeats 3 --uncached 2
    python -m scripts.profile_listing --label after  --photos 3 --repeats 3 --uncached 2

The seller flow is: ``/predict-attributes`` -> ``/search/image`` (similar listings) ->
``/generate-listing`` -> ``/quality-check``. Photos are real catalog photos fetched from the API.
Results go to ``reports/phase9/profile_<label>.json`` and are printed as a table.
The very first request after a server start is reported separately as the COLD run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase9"


def timed(fn: Any) -> tuple[Any, float]:
    """Run ``fn``; return (its result, wall-clock seconds)."""
    start = time.perf_counter()
    res = fn()
    return res, time.perf_counter() - start


def flow(base: str, photo: bytes, attrs: dict[str, Any], notes: dict[str, str]) -> dict[str, float]:
    """One pass of the seller flow; returns wall-clock seconds per stage."""
    files = {"file": ("p.jpg", photo, "image/jpeg")}
    out: dict[str, float] = {}
    r, out["predict_attributes"] = timed(
        lambda: requests.post(f"{base}/predict-attributes", files=files)
    )
    r.raise_for_status()
    pred = r.json()["attributes"]
    _, out["search_image"] = timed(
        lambda: requests.post(f"{base}/search/image", files=files, data={"k": 6}).raise_for_status()
    )
    body = {"attributes": attrs or pred, "seller_notes": notes, "mode": "no_rag_fix"}
    r, out["generate_listing"] = timed(
        lambda: requests.post(f"{base}/generate-listing", json=body, timeout=600)
    )
    r.raise_for_status()
    stated = {k: v["label"] for k, v in (attrs or pred).items()}
    _, out["quality_check"] = timed(
        lambda: requests.post(
            f"{base}/quality-check",
            files=files,
            data={"attributes": json.dumps(stated), "title": "Profile test listing"},
        ).raise_for_status()
    )
    out["total"] = sum(out.values())
    return out


def summarise(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and median seconds per stage."""
    return {
        k: {
            "mean": statistics.mean(r[k] for r in rows),
            "median": statistics.median(r[k] for r in rows),
        }
        for k in rows[0]
    }


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", required=True, help="name of the output file, e.g. before / after")
    ap.add_argument("--photos", type=int, default=3, help="demo products to use")
    ap.add_argument("--repeats", type=int, default=3, help="warm repeats of the cached flow")
    ap.add_argument(
        "--uncached", type=int, default=2, help="flows with a NEW listing (LLM really runs)"
    )
    args = ap.parse_args()
    base = args.url.rstrip("/")

    demo = requests.get(f"{base}/demo/products", timeout=30).json()[: args.photos]
    photos = [(d, requests.get(f"{base}{d['image_url']}", timeout=30).content) for d in demo]

    result: dict[str, Any] = {"label": args.label, "url": base}
    d0, p0 = photos[0]
    cold = flow(base, p0, d0["attributes"], d0["seller_notes"])
    result["cold_first_flow_s"] = cold
    print(f"COLD first flow: {cold}")

    warm = [
        flow(base, p, d["attributes"], d["seller_notes"])
        for _ in range(args.repeats)
        for d, p in photos
    ]
    result["warm_cached_listing"] = summarise(warm)

    fresh = []
    for i in range(args.uncached):
        d, p = photos[i % len(photos)]
        notes = {
            **d["seller_notes"],
            "brand": f"Profile{uuid.uuid4().hex[:6]}",
        }  # unique: cache miss
        fresh.append(flow(base, p, d["attributes"], notes))
    if fresh:
        result["new_listing_uncached"] = summarise(fresh)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"profile_{args.label}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\n{'stage':20s} {'warm (cached listing)':>24s} {'new listing (LLM runs)':>24s}")
    for stage in warm[0]:
        w = result["warm_cached_listing"][stage]["median"]
        n = result.get("new_listing_uncached", {}).get(stage, {}).get("median", float("nan"))
        print(f"{stage:20s} {w:21.3f} s {n:21.3f} s")


if __name__ == "__main__":
    main()
