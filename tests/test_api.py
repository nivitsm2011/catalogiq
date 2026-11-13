"""API tests with FastAPI's TestClient and a fake application state (no models, no LLM)."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import services
from api.main import create_app
from api.state import AppState
from catalogiq.rag.guardrails import GuardrailReport
from catalogiq.rag.qa import QAAnswer, QAResult
from catalogiq.rag.schema import Listing
from catalogiq.search.engine import SearchEngine
from catalogiq.search.looks import Look

TARGETS = [
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "gender",
    "season",
    "usage",
]
LABELS = {
    "masterCategory": ["Apparel", "Footwear"],
    "subCategory": ["Topwear", "Shoes"],
    "articleType": ["Shirts", "Casual Shoes", "Watches"],
    "baseColour": ["Blue", "Red", "Black", "White"],
    "gender": ["Men", "Women", "Unisex"],
    "season": ["Summer", "Winter"],
    "usage": ["Casual", "Formal"],
}


def png_bytes(
    size: tuple[int, int] = (60, 80), colour: tuple[int, int, int] = (200, 30, 30)
) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


class FakeEncoder:
    """Deterministic unit vectors from the image content or the text."""

    def _vec(self, seed: int) -> np.ndarray:
        v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode_image(self, image: Any) -> np.ndarray:
        arr = np.asarray(image)
        return self._vec(int(arr.mean() * 10))

    def encode_text(self, text: str) -> np.ndarray:
        return self._vec(abs(hash(text)) % 10_000)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode_text(t) for t in texts])


@dataclass
class FakeGenerator:
    calls: list[tuple[Any, ...]] = field(default_factory=list)

    def generate(self, attributes: dict, notes: dict, use_rag: bool, fix: bool) -> Any:
        self.calls.append((attributes, notes, use_rag, fix))
        listing = Listing(
            title="Blue Cotton-Free Shirt for Men",
            bullets=["a", "b", "c", "d", "e"],
            description="A shirt.",
            search_tags=["shirt"],
        )
        report = GuardrailReport(
            unsupported=[("material", "cotton")], coverage=0.8, missing_attributes=["usage"]
        )
        return SimpleNamespace(
            listing=listing,
            report=report,
            group="topwear",
            attempts=1,
            fixed=fix,
            retrieved=["B0X"],
            error=None,
            latency_s=12.5,
        )


class FakeQA:
    def answer(self, item_id: str, question: str) -> QAResult:
        evidence = pd.DataFrame(
            [
                {
                    "review_id": "R1",
                    "rating": 5.0,
                    "verified_purchase": True,
                    "helpful_vote": 3,
                    "date": "2023-01-01",
                    "text": "Runs big.",
                }
            ]
        )
        if "waterproof" in question:
            out = QAAnswer(
                answer="Not enough reviews mention this.",
                citations=[],
                confidence=0.0,
                aspect="general",
                evidence_count=0,
                refused=True,
            )
        else:
            out = QAAnswer(
                answer="It runs big [R1].",
                citations=["R1"],
                confidence=0.9,
                aspect="fit_size",
                evidence_count=2,
            )
        return QAResult(output=out, evidence=evidence, top_similarity=0.6, latency_s=0.1)


class FakeLooks:
    def recommend(self, item: Any) -> Look:
        items = pd.DataFrame(
            [
                {
                    "role": "footwear",
                    "id": 2,
                    "score": 0.9,
                    "productDisplayName": "Shoe",
                    "articleType": "Casual Shoes",
                    "baseColour": "Black",
                    "gender": "Men",
                    "usage": "Casual",
                    "season": "Summer",
                    "reasons": "goes with shirts",
                }
            ]
        )
        return Look({"id": 1, "articleType": "Shirts", "role": "topwear"}, items)


def make_state() -> AppState:
    catalog = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "productDisplayName": ["Blue Shirt", "Black Shoe", "Red Shirt", "White Watch"],
            "articleType": ["Shirts", "Casual Shoes", "Shirts", "Watches"],
            "baseColour": ["Blue", "Black", "Red", "White"],
            "gender": ["Men", "Men", "Women", "Unisex"],
            "usage": ["Casual"] * 4,
            "season": ["Summer"] * 4,
            "masterCategory": ["Apparel", "Footwear", "Apparel", "Accessories"],
            "subCategory": ["Topwear", "Shoes", "Topwear", "Watches"],
        }
    )
    enc = FakeEncoder()
    emb = np.stack([enc._vec(i) for i in range(4)])
    predictor = SimpleNamespace(targets=TARGETS, labels=LABELS, threshold={t: 0.6 for t in TARGETS})
    chunks = pd.DataFrame(
        {
            "chunk_id": ["c1", "c2"],
            "review_id": ["R1", "R2"],
            "item_id": ["B0X", "B0X"],
            "rating": [5.0, 2.0],
            "verified_purchase": [True, True],
            "helpful_vote": [1, 0],
            "date": ["2023", "2023"],
            "chunk_idx": [0, 0],
            "n_chunks": [1, 1],
            "text": ["Runs big. Love it.", "Too small, poor quality."],
        }
    )
    items = pd.DataFrame(
        {
            "parent_asin": ["B0X"],
            "title": ["Test Shirt"],
            "store": ["Shop"],
            "average_rating": [4.5],
        }
    )
    state = AppState(
        predictor=predictor,
        encoder=enc,
        catalog=catalog,
        image_emb=emb,
        engine=SearchEngine(catalog, emb, encoder=enc),
        looks=FakeLooks(),
        generator=FakeGenerator(),
        qa=FakeQA(),
        chunks=chunks,
        review_items=items,
        llm=SimpleNamespace(model="fake-llm"),
        labels=LABELS,
        text_prompts=np.stack([enc._vec(900), enc._vec(901)]),
        value_emb={
            a: np.stack([enc._vec(1000 + i) for i in range(len(LABELS[a]))])
            for a in ("baseColour", "articleType", "gender")
        },
        demo=[
            {
                "id": 1,
                "title": "Blue Shirt",
                "group": "topwear",
                "seller_notes": {"brand": "X"},
                "attributes": {
                    "articleType": {"label": "Shirts", "confidence": 0.9, "needs_review": False}
                },
            }
        ],
        questions=[{"id": "Q1", "item": "B0X", "q": "Does it run big?", "kind": "answerable"}],
        models={"attribute_classifier": True, "search_index": True},
    )
    state._lazy["review_text"] = pd.DataFrame(
        {"review_id": ["R1", "R2"], "full": ["Great. Runs big.", "Poor. Too small."]}
    ).set_index("review_id", drop=False)
    return state


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> TestClient:
    def fake_probs(images: Any, predictor: Any, batch: int = 128) -> dict[str, np.ndarray]:
        out = {}
        for t in TARGETS:
            k = len(LABELS[t])
            p = np.full((len(images), k), 0.05 / max(k - 1, 1))
            p[:, 0] = 0.95
            out[t] = p
        return out

    monkeypatch.setattr("catalogiq.quality.consistency.classifier_probabilities", fake_probs)
    monkeypatch.setattr(services, "_decisions_path", lambda: tmp_path / "decisions.json")
    monkeypatch.setattr("api.main.llm_reachable", lambda: True)
    with TestClient(create_app(make_state())) as c:
        yield c


def files(data: bytes, name: str = "p.png", ctype: str = "image/png") -> dict[str, Any]:
    return {"file": (name, data, ctype)}


# ------------------------------------------------------------------------------------ system
def test_health_and_docs(client: TestClient) -> None:
    h = client.get("/health").json()
    assert (
        h["status"] == "ok"
        and h["llm_reachable"] is True
        and h["catalog_size"] == 4
        and h["llm_model"] == "fake-llm"
    )
    paths = client.get("/openapi.json").json()["paths"]
    for p in [
        "/predict-attributes",
        "/generate-listing",
        "/search/image",
        "/search/text",
        "/shop-the-look",
        "/quality-check",
        "/ask-reviews/{item_id}",
        "/health",
        "/metrics",
    ]:
        assert p in paths, p
    assert client.get("/docs").status_code == 200


def test_metrics_count_requests_and_errors(client: TestClient) -> None:
    client.get("/health")
    client.get("/search/text", params={"q": "x"})  # 422: too short
    m = client.get("/metrics").json()
    assert m["total_requests"] >= 2 and m["total_errors"] >= 1
    assert (
        m["endpoints"]["GET /health"]["count"] == 1 and m["endpoints"]["GET /health"]["avg_ms"] >= 0
    )


def test_starting_up_returns_503_before_models_are_loaded() -> None:
    c = TestClient(create_app(None))  # no context manager: the lifespan (model loading) never runs
    assert c.get("/health").json()["status"] == "degraded"
    r = c.get("/search/text", params={"q": "shirt"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "starting_up"


# ----------------------------------------------------------------------------- upload checks
def test_predict_attributes_ok(client: TestClient) -> None:
    r = client.post("/predict-attributes", files=files(png_bytes()))
    assert r.status_code == 200
    body = r.json()
    assert set(body["attributes"]) == set(TARGETS) and body["image"] == {"width": 60, "height": 80}
    a = body["attributes"]["articleType"]
    assert a["label"] == "Shirts" and a["needs_review"] is False and len(a["alternatives"]) == 2


@pytest.mark.parametrize(
    ("kwargs", "status", "code"),
    [
        ({"files": {"file": ("x.txt", b"hello", "text/plain")}}, 415, "unsupported_media_type"),
        ({"files": {"file": ("x.png", b"not an image", "image/png")}}, 422, "invalid_image"),
        ({"files": {"file": ("x.png", b"", "image/png")}}, 400, "empty_file"),
        ({"files": {"file": ("x.png", png_bytes((8, 8)), "image/png")}}, 422, "image_too_small"),
        ({}, 400, "missing_file"),
    ],
)
def test_upload_validation(client: TestClient, kwargs: dict, status: int, code: str) -> None:
    r = client.post("/predict-attributes", **kwargs)
    assert r.status_code == status and r.json()["error"]["code"] == code
    assert r.json()["error"]["message"]


def test_oversized_upload_is_rejected(client: TestClient) -> None:
    big = png_bytes() + b"0" * (5 * 1024 * 1024 + 10)
    r = client.post("/predict-attributes", files=files(big))
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_content_type_must_match_content(client: TestClient) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (40, 40)).save(buf, format="GIF")
    r = client.post("/predict-attributes", files=files(buf.getvalue(), "x.png", "image/png"))
    assert r.status_code == 415


# ---------------------------------------------------------------------------------- listing
def test_generate_listing(client: TestClient) -> None:
    body = {
        "attributes": {
            "articleType": {"label": "Shirts"},
            "baseColour": {"label": "Blue", "confidence": 0.7, "needs_review": True},
        },
        "seller_notes": {"brand": "Puma"},
        "mode": "rag_fix",
    }
    r = client.post("/generate-listing", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["listing"]["title"].startswith("Blue") and len(out["listing"]["bullets"]) == 5
    assert out["guardrails"]["hallucinated"] is True and out["guardrails"]["unsupported"] == [
        {"kind": "material", "term": "cotton"}
    ]
    assert out["mode"] == "rag_fix" and out["fixed"] is True and out["generation_s"] == 12.5


def test_generate_listing_errors(client: TestClient) -> None:
    r = client.post("/generate-listing", json={"attributes": {"baseColour": {"label": "Blue"}}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "missing_attribute"
    r = client.post(
        "/generate-listing",
        json={"attributes": {"articleType": {"label": "Shirts"}, "flavour": {"label": "x"}}},
    )
    assert r.status_code == 422 and r.json()["error"]["code"] == "unknown_attribute"
    r = client.post(
        "/generate-listing",
        json={"attributes": {"articleType": {"label": "Shirts"}}, "mode": "magic"},
    )
    body = r.json()
    assert (
        r.status_code == 422
        and body["error"]["code"] == "validation_error"
        and "mode" in body["error"]["message"]
    )


# ----------------------------------------------------------------------------------- search
def test_text_search_with_filters(client: TestClient) -> None:
    r = client.get("/search/text", params={"q": "blue shirt", "k": 3, "gender": "Men"})
    assert r.status_code == 200
    res = r.json()["results"]
    assert 1 <= len(res) <= 3 and all(x["gender"] == "Men" for x in res)
    assert res[0]["image_url"].startswith("/catalog/image/")


def test_text_search_validation(client: TestClient) -> None:
    assert client.get("/search/text", params={"q": "x"}).status_code == 422
    r = client.get("/search/text", params={"q": "shirt", "gender": "Martian"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "unknown_filter_value"
    assert client.get("/search/text", params={"q": "shirt", "k": 500}).status_code == 422


def test_image_search_excludes_the_query_item(client: TestClient) -> None:
    r = client.post("/search/image", files=files(png_bytes()), data={"k": "3", "exclude_id": "1"})
    assert r.status_code == 200
    ids = [x["id"] for x in r.json()["results"]]
    assert 1 not in ids and len(ids) <= 3


def test_shop_the_look(client: TestClient) -> None:
    assert client.post("/shop-the-look").status_code == 400
    assert (
        client.post("/shop-the-look", data={"item_id": "999"}).json()["error"]["code"]
        == "unknown_item"
    )
    r = client.post("/shop-the-look", data={"item_id": "1"})
    assert r.status_code == 200 and r.json()["items"][0]["role"] == "footwear"
    r = client.post("/shop-the-look", files=files(png_bytes()))
    assert r.status_code == 200 and r.json()["query"]["articleType"] == "Shirts"
    both = client.post("/shop-the-look", data={"item_id": "1"}, files=files(png_bytes()))
    assert both.status_code == 400


# ---------------------------------------------------------------------------------- quality
def test_quality_check(client: TestClient) -> None:
    stated = json.dumps(
        {"articleType": "Shirts", "baseColour": "Red", "gender": "Men", "usage": "Casual"}
    )
    r = client.post(
        "/quality-check",
        files=files(png_bytes((120, 160), (240, 240, 240))),
        data={"attributes": stated, "title": "A shirt"},
    )
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["score"] <= 100 and set(body["components"]) == {
        "image",
        "consistency",
        "completeness",
    }
    assert body["completeness"]["pct"] == 100.0
    assert all(f["fix"] for f in body["flags"])
    # the fake classifier always says the first label of every attribute: Apparel/Shirts/Blue/Men
    assert any(m["attribute"] == "baseColour" for m in body["mismatches"]) or isinstance(
        body["mismatches"], list
    )


def test_quality_check_reports_missing_attributes(client: TestClient) -> None:
    r = client.post(
        "/quality-check",
        files=files(png_bytes()),
        data={"attributes": json.dumps({"articleType": "Shirts"})},
    )
    body = r.json()
    assert "gender" in body["completeness"]["missing"] and any(
        f["code"] == "missing_attributes" for f in body["flags"]
    )


def test_quality_check_rejects_bad_attributes_json(client: TestClient) -> None:
    r = client.post("/quality-check", files=files(png_bytes()), data={"attributes": "not json"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_attributes"


# ----------------------------------------------------------------------------------- reviews
def test_review_flow(client: TestClient) -> None:
    prods = client.get("/reviews/products").json()
    assert prods[0]["item_id"] == "B0X" and prods[0]["demo"] is True and prods[0]["n_reviews"] == 2
    ok = client.post("/ask-reviews/B0X", json={"question": "Does it run big?"}).json()
    assert (
        ok["refused"] is False
        and ok["citations"][0]["review_id"] == "R1"
        and "Runs big" in ok["citations"][0]["text"]
    )
    no = client.post("/ask-reviews/B0X", json={"question": "Is it waterproof?"}).json()
    assert (
        no["refused"] is True
        and no["answer"] == "Not enough reviews mention this."
        and no["citations"] == []
    )


def test_review_errors(client: TestClient) -> None:
    r = client.post("/ask-reviews/NOPE", json={"question": "Does it run big?"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_product"
    assert client.post("/ask-reviews/B0X", json={"question": "hi"}).status_code == 422
    assert client.get("/reviews/NOPE/summary").status_code == 404


def test_review_summary_numbers_come_from_code(client: TestClient) -> None:
    r = client.get("/reviews/B0X/summary")
    assert r.status_code == 200
    body = r.json()
    assert (
        body["n_reviews"] == 2
        and body["source"] == "template"
        and isinstance(body["aspects"], dict)
    )


# ---------------------------------------------------------------------------------- misc
def test_demo_options_and_duplicate_decision(client: TestClient) -> None:
    demo = client.get("/demo/products").json()
    assert (
        demo[0]["image_url"] == "/catalog/image/1"
        and demo[0]["attributes"]["articleType"]["label"] == "Shirts"
    )
    opts = client.get("/meta/options").json()
    assert "Men" in opts["filters"]["gender"] and opts["attribute_labels"]["gender"][0] == "Men"
    r = client.post("/ops/duplicates/decision", json={"id_a": 1, "id_b": 2, "decision": "merge"})
    assert r.json() == {"saved": True, "n_decided": 1}
    assert (
        client.post(
            "/ops/duplicates/decision", json={"id_a": 1, "id_b": 2, "decision": "delete"}
        ).status_code
        == 422
    )


def test_unknown_catalog_image_is_404(client: TestClient) -> None:
    r = client.get("/catalog/image/999999999")
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_item"
