"""Write the three small Lottie animations of the Streamlit app (local files, no download).

Usage: python scripts/make_lottie.py
Outputs: app/assets/loader.json (bouncing dots), scan.json (scanner line), success.json (check mark)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parents[1] / "app" / "assets"
INDIGO, VIOLET, CORAL, TEAL = (
    [0.357, 0.357, 0.941, 1],
    [0.545, 0.361, 0.965, 1],
    [1, 0.42, 0.42, 1],
    [0.078, 0.722, 0.651, 1],
)
EASE = {"i": {"x": [0.4], "y": [1]}, "o": {"x": [0.6], "y": [0]}}


def anim(frames: list[tuple[int, list[float]]]) -> dict[str, Any]:
    """An animated property from (frame, value) keyframes."""
    keys = []
    for n, (t, v) in enumerate(frames):
        key: dict[str, Any] = {"t": t, "s": v}
        if n < len(frames) - 1:
            key.update(
                {
                    "i": {"x": [0.4] * len(v), "y": [1] * len(v)},
                    "o": {"x": [0.6] * len(v), "y": [0] * len(v)},
                }
            )
        keys.append(key)
    return {"a": 1, "k": keys}


def static(v: Any) -> dict[str, Any]:
    return {"a": 0, "k": v}


def transform(pos: Any, scale: Any, opacity: Any = None) -> dict[str, Any]:
    return {
        "o": opacity or static(100),
        "r": static(0),
        "p": pos,
        "a": static([0, 0, 0]),
        "s": scale,
    }


def layer(index: int, shapes: list[dict[str, Any]], ks: dict[str, Any], op: int) -> dict[str, Any]:
    return {
        "ddd": 0,
        "ind": index,
        "ty": 4,
        "nm": f"layer {index}",
        "sr": 1,
        "ks": ks,
        "ao": 0,
        "shapes": shapes,
        "ip": 0,
        "op": op,
        "st": 0,
        "bm": 0,
    }


def ellipse(size: float, colour: list[float]) -> dict[str, Any]:
    return {
        "ty": "gr",
        "it": [
            {"ty": "el", "p": static([0, 0]), "s": static([size, size]), "nm": "ellipse"},
            {"ty": "fl", "c": static(colour), "o": static(100), "r": 1, "nm": "fill"},
            {
                "ty": "tr",
                "p": static([0, 0]),
                "a": static([0, 0]),
                "s": static([100, 100]),
                "r": static(0),
                "o": static(100),
                "sk": static(0),
                "sa": static(0),
            },
        ],
        "nm": "dot",
    }


def wrap(
    name: str, layers: list[dict[str, Any]], frames: int = 60, size: int = 200
) -> dict[str, Any]:
    return {
        "v": "5.7.0",
        "fr": 30,
        "ip": 0,
        "op": frames,
        "w": size,
        "h": size,
        "nm": name,
        "ddd": 0,
        "assets": [],
        "layers": layers,
    }


def loader() -> dict[str, Any]:
    """Three dots that bounce one after another."""
    layers = []
    for n, (x, colour) in enumerate(((60, INDIGO), (100, VIOLET), (140, CORAL))):
        d = n * 6
        y = anim(
            [(0 + d, [x, 110, 0]), (10 + d, [x, 80, 0]), (20 + d, [x, 110, 0]), (60, [x, 110, 0])]
        )
        s = anim(
            [
                (0 + d, [100, 100, 100]),
                (10 + d, [125, 125, 100]),
                (20 + d, [100, 100, 100]),
                (60, [100, 100, 100]),
            ]
        )
        layers.append(layer(n + 1, [ellipse(30, colour)], transform(y, s), 60))
    return wrap("loader", layers)


def scan() -> dict[str, Any]:
    """A rounded frame with a scanner line sweeping down and up."""
    frame = {
        "ty": "gr",
        "it": [
            {
                "ty": "rc",
                "d": 1,
                "s": static([110, 130]),
                "p": static([0, 0]),
                "r": static(14),
                "nm": "frame",
            },
            {
                "ty": "st",
                "c": static(INDIGO),
                "o": static(100),
                "w": static(6),
                "lc": 2,
                "lj": 2,
                "nm": "stroke",
            },
            {
                "ty": "tr",
                "p": static([0, 0]),
                "a": static([0, 0]),
                "s": static([100, 100]),
                "r": static(0),
                "o": static(100),
                "sk": static(0),
                "sa": static(0),
            },
        ],
        "nm": "frame",
    }
    line = {
        "ty": "gr",
        "it": [
            {
                "ty": "rc",
                "d": 1,
                "s": static([96, 7]),
                "p": static([0, 0]),
                "r": static(3),
                "nm": "line",
            },
            {"ty": "fl", "c": static(CORAL), "o": static(100), "r": 1, "nm": "fill"},
            {
                "ty": "tr",
                "p": static([0, 0]),
                "a": static([0, 0]),
                "s": static([100, 100]),
                "r": static(0),
                "o": static(100),
                "sk": static(0),
                "sa": static(0),
            },
        ],
        "nm": "scan line",
    }
    y = anim([(0, [100, 55, 0]), (30, [100, 145, 0]), (60, [100, 55, 0])])
    return wrap(
        "scan",
        [
            layer(1, [line], transform(y, static([100, 100, 100])), 60),
            layer(2, [frame], transform(static([100, 100, 0]), static([100, 100, 100])), 60),
        ],
    )


def success() -> dict[str, Any]:
    """A circle that pops in with a check mark drawn inside."""
    circle = ellipse(120, TEAL)
    pop = anim(
        [(0, [0, 0, 100]), (14, [112, 112, 100]), (22, [100, 100, 100]), (60, [100, 100, 100])]
    )
    check = {
        "ty": "gr",
        "it": [
            {
                "ty": "sh",
                "ks": static(
                    {
                        "i": [[0, 0], [0, 0], [0, 0]],
                        "o": [[0, 0], [0, 0], [0, 0]],
                        "v": [[-24, 2], [-7, 20], [26, -18]],
                        "c": False,
                    }
                ),
                "nm": "path",
            },
            {
                "ty": "st",
                "c": static([1, 1, 1, 1]),
                "o": static(100),
                "w": static(11),
                "lc": 2,
                "lj": 2,
                "nm": "stroke",
            },
            {
                "ty": "tm",
                "s": static(0),
                "e": anim([(14, [0]), (34, [100]), (60, [100])]),
                "o": static(0),
                "m": 1,
                "nm": "trim",
            },
            {
                "ty": "tr",
                "p": static([0, 0]),
                "a": static([0, 0]),
                "s": static([100, 100]),
                "r": static(0),
                "o": static(100),
                "sk": static(0),
                "sa": static(0),
            },
        ],
        "nm": "check",
    }
    return wrap(
        "success",
        [
            layer(1, [check], transform(static([100, 100, 0]), pop), 60),
            layer(2, [circle], transform(static([100, 100, 0]), pop), 60),
        ],
    )


def main() -> None:
    """Write the animation files."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("loader", loader), ("scan", scan), ("success", success)):
        (OUT / f"{name}.json").write_text(json.dumps(fn(), separators=(",", ":")), encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.glob("*.json")))


if __name__ == "__main__":
    main()
