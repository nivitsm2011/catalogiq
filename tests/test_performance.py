"""Phase 9 performance shortcuts: per-photo memo, warm-up, model keep-alive."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import services
from api.main import create_app
from api.state import PHOTO_CACHE_SIZE
from catalogiq.llm.client import CachedClient, OllamaClient
from tests.test_api import LABELS, TARGETS, FakeEncoder, files, make_state, png_bytes


@pytest.fixture()
def api_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    def fake_probs(images: Any, predictor: Any, batch: int = 128) -> dict[str, np.ndarray]:
        out = {}
        for t in TARGETS:
            k = len(LABELS[t])
            p = np.full((len(images), k), 0.05 / max(k - 1, 1))
            p[:, 0] = 0.95
            out[t] = p
        return out

    monkeypatch.setattr("catalogiq.quality.consistency.classifier_probabilities", fake_probs)
    monkeypatch.setattr("api.main.llm_reachable", lambda: True)
    with TestClient(create_app(make_state())) as c:
        yield c


def test_photo_memo_computes_once_per_photo_and_evicts_the_oldest() -> None:
    state = make_state()
    calls: list[int] = []

    def compute() -> int:
        calls.append(1)
        return len(calls)

    a = np.zeros((8, 8, 3), dtype=np.uint8)
    b = a.copy()
    b[0, 0, 0] = 1
    assert state.photo_memo("clip", a, compute) == 1
    assert state.photo_memo("clip", a.copy(), compute) == 1  # same content: cached
    assert state.photo_memo("clf", a, compute) == 2  # another kind of result: separate entry
    assert state.photo_memo("clip", b, compute) == 3  # different content: separate entry
    for i in range(PHOTO_CACHE_SIZE + 2):  # fill the cache: the first photo is evicted
        state.photo_memo("clip", np.full((8, 8, 3), i + 10, dtype=np.uint8), compute)
    before = len(calls)
    state.photo_memo("clip", a, compute)
    assert len(calls) == before + 1


class CountingEncoder(FakeEncoder):
    def __init__(self) -> None:
        self.images = 0

    def encode_image(self, image: Any) -> np.ndarray:
        self.images += 1
        return super().encode_image(image)


def test_one_photo_is_embedded_once_across_search_and_quality_check(
    api_client: TestClient,
) -> None:
    state = api_client.app.state.catalogiq
    encoder = CountingEncoder()
    state.encoder = encoder
    photo = png_bytes((60, 80), (30, 70, 200))
    assert api_client.post("/search/image", files=files(photo)).status_code == 200
    assert encoder.images == 1
    r = api_client.post("/quality-check", files=files(photo), data={"attributes": "{}"})
    assert r.status_code == 200 and encoder.images == 1  # the embedding was reused
    other = png_bytes((60, 80), (200, 30, 30))
    api_client.post("/quality-check", files=files(other), data={"attributes": "{}"})
    assert encoder.images == 2  # a different photo is embedded again


def test_cached_embedding_gives_the_same_search_results(api_client: TestClient) -> None:
    photo = png_bytes((60, 80), (30, 70, 200))
    first = api_client.post("/search/image", files=files(photo)).json()["results"]
    second = api_client.post("/search/image", files=files(photo)).json()["results"]
    assert [r["id"] for r in first] == [r["id"] for r in second]


def test_warm_up_touches_the_models_and_skips_the_reply_cache() -> None:
    state = make_state()
    state.predictor = SimpleNamespace(
        targets=["gender"],
        temperature={"gender": 1.0},
        preprocess=lambda im: __import__("torch").zeros(1, 3, 4, 4),
        model=lambda x: [__import__("torch").zeros(1, 2)],
    )
    encoder = CountingEncoder()
    state.encoder = encoder
    asked: list[Any] = []
    inner = SimpleNamespace(chat=lambda msgs, **kw: asked.append(msgs))
    state.llm = SimpleNamespace(inner=inner, model="m")
    state.generator = SimpleNamespace(
        retriever=SimpleNamespace(search=lambda *a, **k: None),
    )
    services.warm_up(state)
    assert encoder.images == 1 and len(asked) == 1  # the model was called directly, not the cache
    assert {"clf"} <= {k[0] for k in state._photo_cache}


def test_warm_up_llm_failure_does_not_stop_the_other_steps() -> None:
    state = make_state()

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("ollama is down")

    state.llm = SimpleNamespace(chat=boom, model="m")
    encoder = CountingEncoder()
    state.encoder = encoder
    state.predictor = SimpleNamespace(targets=[])
    services.warm_up(state)  # must not raise
    assert encoder.images == 1


def test_ollama_client_passes_keep_alive_and_cache_wraps_unchanged(tmp_path: Any) -> None:
    seen: dict[str, Any] = {}

    def chat(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return SimpleNamespace(message=SimpleNamespace(content="{}"), eval_count=3)

    client = OllamaClient("m", "http://localhost:11434", 5.0, 0.2, 1, keep_alive="30m")
    client._client = SimpleNamespace(chat=chat)  # type: ignore[assignment]
    cached = CachedClient(client, tmp_path)
    reply = cached.chat([{"role": "user", "content": "hi"}])
    assert seen["keep_alive"] == "30m" and reply.output_tokens == 3 and not reply.cached
    assert cached.chat([{"role": "user", "content": "hi"}]).cached  # second call from disk


def test_configured_keep_alive_is_longer_than_ollamas_default() -> None:
    from catalogiq.config import get_config

    value = get_config()["models"]["llm"]["keep_alive"]
    assert value.endswith("m") and int(value[:-1]) > 5


def test_photo_shrink_keeps_small_photos_and_scales_large_ones() -> None:
    small = Image.new("RGB", (60, 80))
    assert services.shrink(small) is small
    big = Image.new("RGB", (2000, 1000))
    shrunk = services.shrink(big)
    assert max(shrunk.size) == services.MAX_MODEL_SIDE and shrunk.size == (1024, 512)


@pytest.mark.parametrize("size", [(60, 80), (300, 400)])
def test_listing_endpoint_still_works_with_the_memo(
    api_client: TestClient,
    size: tuple[int, int],
) -> None:
    r = api_client.post("/predict-attributes", files=files(png_bytes(size)))
    assert r.status_code == 200 and set(r.json()["attributes"]) == set(TARGETS)
