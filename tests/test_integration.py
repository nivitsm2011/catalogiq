"""Photo -> attributes -> listing -> quality, with a tiny real classifier and a mocked LLM.

Only the heavy parts are replaced: the language model (canned replies) and CLIP (a fake encoder).
The classifier is a real exported TorchScript network (random weights), the image checks, the
guardrails, the listing generator, the quality score and the API layer are the real code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from catalogiq.llm.client import LLMClient, LLMResponse
from catalogiq.quality.image_checks import check_image
from catalogiq.rag.listing import ListingGenerator
from catalogiq.vision.predict import AttributePredictor
from tests.conftest import load_rgb
from tests.test_api import make_state
from tests.test_rag import good_listing


class RecordingLLM(LLMClient):
    """Always answers with one canned listing; keeps every prompt it was given."""

    model = "mock-llm"

    def __init__(self) -> None:
        self.prompts: list[list[dict[str, str]]] = []

    def chat(self, messages, schema=None, temperature=None, max_tokens=None) -> LLMResponse:
        self.prompts.append(messages)
        return LLMResponse(json.dumps(good_listing()), 0.01)


@pytest.fixture()
def pipeline(tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch):
    llm = RecordingLLM()
    state = make_state()
    predictor = AttributePredictor(tiny_model_dir)
    state.predictor = predictor
    state.labels = {t: predictor.labels[t] for t in predictor.targets}
    state.generator = ListingGenerator(llm, None, ["puma"])
    monkeypatch.setattr("api.main.llm_reachable", lambda: True)
    with TestClient(create_app(state)) as client:
        yield client, llm, state


def png(path: Path) -> dict[str, Any]:
    return {"file": (path.name, path.read_bytes(), "image/jpeg")}


def test_photo_to_listing_to_quality(pipeline, photo_dir: Path) -> None:
    client, llm, _ = pipeline
    photo = photo_dir / "shirt_blue.jpg"

    # 1. photo -> attributes (real classifier)
    r = client.post("/predict-attributes", files=png(photo))
    assert r.status_code == 200
    attrs = r.json()["attributes"]
    assert set(attrs) == {"articleType", "gender"}
    assert all(a["needs_review"] for a in attrs.values())  # a random network is never 99% sure

    # 2. the seller confirms the attributes -> listing (mocked LLM, real prompt + guardrails)
    body = {"attributes": attrs, "seller_notes": {"brand": "Puma"}, "mode": "no_rag_fix"}
    r = client.post("/generate-listing", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["listing"]["title"] and len(out["listing"]["bullets"]) == 5
    assert out["guardrails"]["coverage"] >= 0 and out["error"] is None
    prompt = llm.prompts[0][1]["content"]
    assert attrs["articleType"]["label"] in prompt and "UNCERTAIN" in prompt
    assert "Puma" in prompt  # the seller's note reaches the model

    # 3. the same photo and the new title -> quality score with plain-English fixes
    stated = {k: v["label"] for k, v in attrs.items()}
    r = client.post(
        "/quality-check",
        files=png(photo),
        data={"attributes": json.dumps(stated), "title": out["listing"]["title"]},
    )
    assert r.status_code == 200
    quality = r.json()
    assert 0 <= quality["score"] <= 100 and "components" in quality
    assert quality["image_issues"] == []  # the clean fixture photo raises no image flag


def test_unsupported_claims_in_the_model_reply_are_caught(pipeline, photo_dir: Path) -> None:
    client, llm, state = pipeline
    bad = good_listing(bullets=["100% cotton", "Waterproof", "Easy care", "Slim fit", "Soft"])
    llm.chat = lambda *a, **k: LLMResponse(json.dumps(bad), 0.01)  # type: ignore[method-assign]
    attrs = {"articleType": {"label": "Tshirts", "confidence": 0.9, "needs_review": False}}
    r = client.post("/generate-listing", json={"attributes": attrs, "mode": "no_rag"})
    guard = r.json()["guardrails"]
    assert (
        guard["hallucinated"] and guard["unsupported"]
    )  # invented material and claims are flagged


def test_a_blurry_photo_lowers_the_score_and_explains_how_to_fix_it(pipeline, photo_dir) -> None:
    client, _, _ = pipeline
    stated = json.dumps({"articleType": "Tshirts"})
    good = client.post(
        "/quality-check", files=png(photo_dir / "shirt_blue.jpg"), data={"attributes": stated}
    ).json()
    bad = client.post(
        "/quality-check", files=png(photo_dir / "blurry.jpg"), data={"attributes": stated}
    ).json()
    assert "blurry" in bad["image_issues"] and bad["score"] < good["score"]
    blur_flags = [f for f in bad["flags"] if f["code"] == "blurry"]
    assert blur_flags and "retake" in blur_flags[0]["fix"].lower()


# ------------------------------------------------------------------------- image checks only
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("shirt_blue.jpg", set()),
        ("shirt_red.png", set()),
        ("shirt_black.webp", set()),
        ("blurry.jpg", {"blurry"}),
        ("dark.jpg", {"too_dark"}),
        ("cluttered.png", {"cluttered_background"}),
        ("tiny.png", {"low_resolution"}),
    ],
)
def test_image_checks_on_fixture_photos(name: str, expected: set[str]) -> None:
    issues = set(check_image(load_rgb(name))["issues"])
    assert expected <= issues  # the injected defect is found
    if not expected:
        assert not issues  # clean photos are never flagged


def test_every_fixture_format_is_accepted_by_the_api(pipeline, photo_dir: Path) -> None:
    client, _, _ = pipeline
    for name, ctype in [
        ("shirt_blue.jpg", "image/jpeg"),
        ("shirt_red.png", "image/png"),
        ("shirt_black.webp", "image/webp"),
    ]:
        r = client.post(
            "/predict-attributes", files={"file": (name, (photo_dir / name).read_bytes(), ctype)}
        )
        assert r.status_code == 200, name
    bad = client.post(
        "/predict-attributes",
        files={"file": ("x.png", (photo_dir / "not_an_image.txt").read_bytes(), "image/png")},
    )
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_image"
    tiny = client.post(
        "/predict-attributes",
        files={"file": ("t.png", (photo_dir / "tiny.png").read_bytes(), "image/png")},
    )
    assert tiny.status_code == 422 and tiny.json()["error"]["code"] == "image_too_small"
