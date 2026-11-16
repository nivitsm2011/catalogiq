"""Regenerate the small committed test fixtures (a few KB in total, deterministic).

Run:  python tests/fixtures/make_fixtures.py
Nothing here is real catalog or review data: photos are drawn shapes, text is invented.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
PHOTOS = HERE / "photos"
SIZE = (60, 80)  # width, height: the catalog's native thumbnail size


def shirt(
    colour: tuple[int, int, int], background: tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """A flat shirt-like shape on a plain background."""
    img = Image.new("RGB", SIZE, background)
    d = ImageDraw.Draw(img)
    d.polygon(
        [
            (8, 18),
            (22, 8),
            (38, 8),
            (52, 18),
            (46, 30),
            (42, 26),
            (42, 70),
            (18, 70),
            (18, 26),
            (14, 30),
        ],
        fill=colour,
    )
    d.polygon([(24, 8), (30, 16), (36, 8)], fill=(235, 235, 235))
    return img


def make_photos() -> None:
    PHOTOS.mkdir(parents=True, exist_ok=True)
    clean = shirt((30, 70, 200))
    clean.save(PHOTOS / "shirt_blue.jpg", quality=85)
    shirt((200, 40, 40)).save(PHOTOS / "shirt_red.png")
    shirt((20, 20, 20)).save(PHOTOS / "shirt_black.webp", quality=80)
    big = shirt((30, 70, 200)).resize((240, 320), Image.LANCZOS)
    big.filter(ImageFilter.GaussianBlur(9)).save(PHOTOS / "blurry.jpg", quality=85)
    dark = (np.asarray(clean).astype(float) * 0.12).astype(np.uint8)
    Image.fromarray(dark).save(PHOTOS / "dark.jpg", quality=85)
    rng = np.random.default_rng(0)
    blocks = rng.integers(
        0, 255, (16, 12, 3), dtype=np.uint8
    )  # coarse random blocks: busy but tiny
    Image.fromarray(blocks).resize(SIZE, Image.NEAREST).save(PHOTOS / "cluttered.png")
    Image.new("RGB", (8, 8), (120, 120, 120)).save(PHOTOS / "tiny.png")
    (PHOTOS / "not_an_image.txt").write_text("this is not a photo\n", encoding="utf-8")


def make_tables() -> None:
    styles = [
        "id,gender,masterCategory,subCategory,articleType,baseColour,season,year,usage,productDisplayName",
        "1001,Men,Apparel,Topwear,Shirts,Blue,Summer,2012,Casual,Arrow Blue Slim Shirt",
        "1002,Men,Apparel,Topwear,Shirts,Navy Blue,Summer,2012,Formal,Shirt, slim fit, cotton",
        "1003,Women,Apparel,Topwear,Tops,Red,Summer,2013,Casual,Red Summer Top",
        "1004,Women,Apparel,Topwear,Tops,Maroon,Winter,2013,Casual,Maroon Knit Top",
        "1005,Men,Footwear,Shoes,Casual Shoes,Black,Fall,2014,Casual,Black Canvas Sneakers",
        "1006,Men,Footwear,Shoes,Casual Shoes,Black,Fall,2014,Casual,Black Canvas Sneakers Pro",
        "1007,Unisex,Accessories,Watches,Watches,Silver,Winter,2015,Casual,Silver Analog Watch",
        "1008,Men,Apparel,Topwear",
    ]
    (HERE / "styles_sample.csv").write_text("\n".join(styles) + "\n", encoding="utf-8")

    reviews = []
    texts = [
        "The dress runs a bit small, order one size up. Lovely fabric though and very soft.",
        "<b>Great</b> value for the price, arrived quickly and the stitching is solid.",
        "Too short",  # dropped: shorter than the minimum length
        "The dress runs a bit small, order one size up. Lovely fabric though and very soft.",
        "Colour is exactly as pictured, comfortable all day, would buy again.",
        "Stitching came apart after two washes. Disappointed with the quality overall.",
    ]
    for i, t in enumerate(texts):
        reviews.append(
            {
                "parent_asin": "B0AAA" if i < 4 else "B0BBB",
                "rating": 5.0 - i % 3,
                "title": f"Review {i}",
                "text": t,
                "helpful_vote": i,
                "verified_purchase": i % 2 == 0,
                "timestamp": 1_600_000_000_000 + i * 86_400_000,
            }
        )
    reviews.append(
        {
            "parent_asin": "B0CCC",
            "rating": 4.0,
            "title": "other",
            "text": "An item we do not keep, long enough to pass.",
            "helpful_vote": 0,
            "verified_purchase": True,
            "timestamp": 1_600_000_000_000,
        }
    )
    (HERE / "reviews_sample.jsonl").write_text(
        "\n".join(json.dumps(r) for r in reviews) + "\n", encoding="utf-8"
    )

    meta = [
        {
            "parent_asin": "B0AAA",
            "title": "<b>Floral</b> Summer Dress",
            "features": ["Soft fabric", "<i>Machine washable</i>", ""],
            "description": ["A light dress.", "Great for summer."],
            "average_rating": 4.4,
            "rating_number": 120,
            "price": "29.99",
            "store": "Dress Co",
            "categories": ["Clothing"],
            "main_category": "AMAZON FASHION",
        },
        {
            "parent_asin": "B0BBB",
            "title": "Canvas Tote",
            "features": None,
            "description": None,
            "average_rating": 3.9,
            "rating_number": 40,
            "price": "None",
            "store": None,
            "categories": None,
            "main_category": "AMAZON FASHION",
        },
        {
            "parent_asin": "B0DDD",
            "title": "",
            "features": [],
            "description": [],
            "average_rating": None,
            "rating_number": None,
            "price": 12,
            "store": "X",
            "categories": [],
            "main_category": None,
        },
    ]
    (HERE / "meta_sample.jsonl").write_text(
        "\n".join(json.dumps(m) for m in meta) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    make_photos()
    make_tables()
    total = sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
    print(f"fixtures written to {HERE} ({total / 1024:.1f} KB)")
