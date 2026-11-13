# ruff: noqa: E501  (JavaScript snippets and long docstrings)
"""Take the app screenshots in docs/screenshots (headless Edge through the DevTools protocol).

Both servers must be running (API on :8000, Streamlit on :8501). Pages are opened through deep links that
pre-select a demo product; a few shots click a tab or ask a question first.
Usage: python scripts/take_screenshots.py [name ...]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from scripts.browser_driver import Browser

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
BASE = "http://localhost:8501"


async def home(b: Browser) -> None:
    await b.goto(BASE + "/", 9)


async def seller(b: Browser, pid: int = 23961) -> None:
    await b.goto(f"{BASE}/Seller_Studio?demo={pid}&autorun=1", 8)
    await b.wait_idle(120)


async def shop_search(b: Browser) -> None:
    await b.goto(BASE + "/Shopper_View?q=black%20formal%20shoes%20for%20men", 8)
    await b.wait_idle(90)
    await b.click_text("View", "button", 0)
    await b.wait_idle(90)


async def shop_reviews(b: Browser) -> None:
    await b.goto(BASE + "/Shopper_View?product=B07CQ84KLT", 8)
    await b.wait_idle(60)
    await b.click_text("Reviews & questions", '[role="tab"]')
    await asyncio.sleep(3)
    await b.wait_idle(90)
    await b.click_text("Is the dress long", "button")
    await b.wait_idle(180)


async def ops(b: Browser) -> None:
    await b.goto(BASE + "/Catalog_Ops", 10)
    await b.wait_idle(180)


async def card(b: Browser) -> None:
    await b.goto(BASE + "/Model_Card", 9)
    await b.wait_idle(60)


SHOTS = {
    "01_home": home,
    "02_seller_studio_shirt": lambda b: seller(b, 23961),
    "03_seller_studio_handbag": lambda b: seller(b, 48178),
    "04_shopper_search_and_outfit": shop_search,
    "05_shopper_reviews_qa": shop_reviews,
    "06_catalog_ops": ops,
    "07_model_card": card,
}


async def main(names: list[str]) -> None:
    for name in names:
        async with Browser() as b:
            await SHOTS[name](b)
            errs = await b.errors()
            await b.screenshot(OUT / f"{name}.png")
            print(f"{name}: saved" + (f" (page errors: {errs})" if errs else ""))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or list(SHOTS)))
