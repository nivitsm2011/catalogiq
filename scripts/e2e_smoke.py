# ruff: noqa: E501  (JavaScript snippets and long docstrings)
"""End-to-end click-through of every app flow in a real (headless Edge) browser.

Needs both servers running (API :8000, Streamlit :8501). Each check opens a page, clicks or types like a user would,
waits for Streamlit to finish, and fails on missing text or on any error box shown on the page.
Usage: python scripts/e2e_smoke.py [flow ...]      flows: home seller_demo seller_upload shopper_search shopper_reviews ops model_card
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from scripts.browser_driver import Browser

BASE = "http://localhost:8501"
ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "data" / "raw" / "fashion" / "images" / "23961.jpg"
results: list[tuple[str, bool, str]] = []


async def check(b: Browser, name: str, must_have: list[str]) -> None:
    missing = await b.wait_for_text(must_have)
    errs = await b.errors()
    ok = not missing and not errs
    results.append((name, ok, "" if ok else f"missing {missing} errors {errs}"))
    print(
        ("PASS " if ok else "FAIL ")
        + name
        + ("" if ok else f"  -> missing {missing} errors {errs[:1]}")
    )


async def home(b: Browser) -> None:
    await b.goto(BASE + "/", 8)
    await check(
        b,
        "home: counters and role cards",
        [
            "Better listings",
            "Seller Studio",
            "Shopper View",
            "Catalog Ops",
            "Measured on held-out test sets",
        ],
    )


async def seller_demo(b: Browser) -> None:
    await b.goto(BASE + "/Seller_Studio", 8)
    await b.click_text("Use this photo", "button", 0)
    await b.wait_idle(60)
    await check(
        b,
        "seller: demo photo -> attributes",
        ["Confirm the attributes", "please confirm", "Similar existing listings"],
    )
    await b.click_text("Generate listing", "button")
    await b.wait_idle(180)
    await check(
        b,
        "seller: demo listing is instant, with fact check and quality",
        ["Your listing", "Fact check", "Quality score", "Download JSON"],
    )


async def seller_upload(b: Browser) -> None:
    await b.goto(BASE + "/Seller_Studio", 8)
    await b.click_text("Upload a photo", '[role="tab"]')
    await asyncio.sleep(1.5)
    ok = await b.upload(PHOTO)
    results.append(("seller: upload accepted", ok, ""))
    await b.wait_idle(60)
    await check(b, "seller: uploaded photo is read", ["Confirm the attributes", "23961.jpg"])
    picked = await b.select_option(
        "Occasion", "Formal"
    )  # the chip must turn into "confirmed by you"
    results.append(("seller: attribute dropdown opened and option picked", picked, ""))
    print(("PASS " if picked else "FAIL ") + "seller: attribute dropdown opened and option picked")
    await b.wait_idle(60)
    await check(b, "seller: edited attribute is marked as confirmed", ["confirmed by you"])
    print("     (generating a NEW listing: this really runs the local LLM, about 15-30 s)")
    await b.click_text("Generate listing", "button")
    await b.wait_idle(300)
    await check(b, "seller: live generation after edit", ["Your listing", "Quality score"])


async def shopper_search(b: Browser) -> None:
    await b.goto(BASE + "/Shopper_View?q=black%20formal%20shoes%20for%20men", 8)
    await b.wait_idle(60)
    await check(b, "shopper: text search results", ["results", "Formal"])
    await b.click_text("View", "button", 0)
    await b.wait_idle(60)
    await check(b, "shopper: product detail with shop-the-look", ["Product", "Shop the look"])


async def shopper_reviews(b: Browser) -> None:
    await b.goto(BASE + "/Shopper_View", 8)
    await b.click_text("Reviews & questions", '[role="tab"]')
    await asyncio.sleep(3)
    await b.wait_idle(60)
    await check(
        b, "reviews: aspect summary bars", ["What reviewers talk about", "instant demo answers"]
    )
    await b.click_text("Is the dress long", "button")
    await b.wait_idle(180)
    await check(
        b,
        "reviews: suggested question answered with citations",
        ["confidence", "reviews used", "Reviewers agree"],
    )
    ok = await b.type_chat("Is it waterproof?")
    results.append(("reviews: typed question accepted", ok, ""))
    await b.wait_idle(300)
    await check(b, "reviews: typed question answered or refused honestly", ["Is it waterproof?"])
    opened = await b.click_text("R", "button", 0)
    results.append(("reviews: citation chip clickable", bool(opened), ""))


async def ops(b: Browser) -> None:
    await b.goto(BASE + "/Catalog_Ops", 10)
    await b.wait_idle(240)
    await check(
        b,
        "ops: KPIs, charts, worst listings",
        ["Worst listings", "Duplicate review queue", "test listings checked"],
    )
    import re

    def decided(text: str) -> int:
        m = re.search(r"(\d+) decided", text)
        return int(m.group(1)) if m else -1

    before = decided(await b.text())
    clicked = await b.click_text("Keep both", "button", 0)
    await b.wait_idle(120)
    await b.wait_for_text([f"{before + 1} decided"], 30)
    after = decided(await b.text())
    ok = clicked and after == before + 1
    results.append(("ops: duplicate decision is saved and counted", ok, f"{before} -> {after}"))
    print(
        ("PASS " if ok else "FAIL ")
        + f"ops: duplicate decision is saved and counted ({before} -> {after})"
    )


async def model_card(b: Browser) -> None:
    await b.goto(BASE + "/Model_Card", 8)
    await b.wait_idle(60)
    await check(
        b,
        "model card: metrics, architecture, scorecard",
        ["Architecture", "Scorecard", "Limitations and risks", "Scorecard generated"],
    )


FLOWS = {
    "home": home,
    "seller_demo": seller_demo,
    "seller_upload": seller_upload,
    "shopper_search": shopper_search,
    "shopper_reviews": shopper_reviews,
    "ops": ops,
    "model_card": model_card,
}


async def main(names: list[str]) -> int:
    for name in names:
        async with Browser() as b:
            try:
                await FLOWS[name](b)
            except Exception as exc:  # noqa: BLE001
                results.append((name, False, repr(exc)))
                print(f"FAIL {name}: {exc!r}")
            if b.console:
                print("     console:", b.console[:2])
    bad = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:] or list(FLOWS))))
