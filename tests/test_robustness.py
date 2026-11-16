"""Phase 9 robustness: Ollama down, missing files, request-size and rate limits."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from api.limits import RateLimiter
from api.main import create_app
from api.preflight import MissingFilesError, check_required_files, missing_files
from catalogiq.config import get_api_config
from catalogiq.llm.client import (
    LLMError,
    LLMModelMissing,
    LLMUnavailable,
    OllamaClient,
    is_unavailable_message,
)
from tests.test_api import FakeGenerator, files, make_state, png_bytes

ATTRS = {"articleType": {"label": "Shirts", "confidence": 0.9, "needs_review": False}}


def config_with(**rate: Any) -> dict[str, Any]:
    cfg = copy.deepcopy(get_api_config())
    cfg["limits"]["rate_limit"].update(rate)
    return cfg


# ------------------------------------------------------------------------------ LLM is down
def test_friendly_messages_are_recognised() -> None:
    down = str(LLMUnavailable("http://localhost:11434"))
    missing = str(LLMModelMissing("llama3.2:3b"))
    assert "ollama serve" in down and "http://localhost:11434" in down
    assert "ollama pull llama3.2:3b" in missing
    assert is_unavailable_message(down) and is_unavailable_message(missing)
    assert not is_unavailable_message("The language model replied with invalid JSON")
    assert not is_unavailable_message(None) and not is_unavailable_message("")


class StatusError(Exception):
    status_code = 404


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (ConnectionError("refused"), LLMUnavailable),
        (httpx.ConnectError("refused"), LLMUnavailable),
        (httpx.ConnectTimeout("slow"), LLMUnavailable),
        (StatusError("model not found"), LLMModelMissing),
        (httpx.ReadTimeout("slow"), LLMError),
        (ValueError("boom"), LLMError),
    ],
)
def test_ollama_errors_are_classified(raised: Exception, expected: type[LLMError]) -> None:
    client = OllamaClient("m", "http://localhost:11434", 1.0, 0.2, 1)

    def boom(**_: Any) -> Any:
        raise raised

    client._client = SimpleNamespace(chat=boom)  # type: ignore[assignment]
    with pytest.raises(LLMError) as err:
        client.chat([{"role": "user", "content": "hi"}])
    assert type(err.value) is expected
    assert str(err.value)


def state_with_llm_error(message: str):
    state = make_state()

    class Down(FakeGenerator):
        def generate(self, attributes: dict, notes: dict, use_rag: bool, fix: bool) -> Any:
            res = super().generate(attributes, notes, use_rag, fix)
            res.listing, res.report, res.error = None, None, message
            return res

    state.generator = Down()
    state.qa = SimpleNamespace(
        answer=lambda item, q: SimpleNamespace(error=message, output=None, evidence=None)
    )
    return state


def test_listing_returns_503_with_instructions_when_ollama_is_down() -> None:
    with TestClient(create_app(state_with_llm_error(str(LLMUnavailable("http://x"))))) as c:
        r = c.post("/generate-listing", json={"attributes": ATTRS})
        assert r.status_code == 503
        err = r.json()["error"]
        assert err["code"] == "llm_unavailable" and "ollama serve" in err["message"]
        q = c.post("/ask-reviews/B0X", json={"question": "Does it run big?"})
        assert q.status_code == 503 and q.json()["error"]["code"] == "llm_unavailable"


def test_other_llm_errors_stay_in_the_payload() -> None:
    with TestClient(create_app(state_with_llm_error("bad JSON from the model"))) as c:
        r = c.post("/generate-listing", json={"attributes": ATTRS})
        assert r.status_code == 200 and r.json()["error"] == "bad JSON from the model"
        assert r.json()["listing"] is None


def test_unhandled_llm_error_maps_to_503_or_502() -> None:
    state = make_state()

    def raise_down(*_: Any, **__: Any) -> Any:
        raise LLMUnavailable("http://x")

    state.generator = SimpleNamespace(generate=raise_down)
    with TestClient(create_app(state), raise_server_exceptions=False) as c:
        assert c.post("/generate-listing", json={"attributes": ATTRS}).status_code == 503

    def raise_other(*_: Any, **__: Any) -> Any:
        raise LLMError("odd failure")

    state.generator = SimpleNamespace(generate=raise_other)
    with TestClient(create_app(state), raise_server_exceptions=False) as c:
        r = c.post("/generate-listing", json={"attributes": ATTRS})
        assert r.status_code == 502 and r.json()["error"]["code"] == "llm_error"


# --------------------------------------------------------------------------- missing files
def test_missing_files_are_listed_with_the_fix(tmp_path: Any) -> None:
    present = tmp_path / "here.txt"
    present.write_text("x")
    required = [
        {"path": str(present), "how": "n/a"},
        {"path": str(tmp_path / "gone.bin"), "how": "run the export script"},
    ]
    assert missing_files(required) == [(str(tmp_path / "gone.bin"), "run the export script")]
    with pytest.raises(MissingFilesError) as err:
        check_required_files(required)
    text = str(err.value)
    assert "gone.bin" in text and "run the export script" in text and "here.txt" not in text
    check_required_files([required[0]])  # nothing missing: no error


def test_configured_required_files_name_a_fix_for_each() -> None:
    for item in get_api_config()["required_files"]:
        assert item["path"] and "python" in item["how"]


# -------------------------------------------------------------------------------- limits
def test_rate_limiter_sliding_window() -> None:
    now = [0.0]
    limiter = RateLimiter(10.0, clock=lambda: now[0])
    assert limiter.allow("a", "all", 2) == (True, 0.0)
    assert limiter.allow("a", "all", 2)[0] is True
    allowed, wait = limiter.allow("a", "all", 2)
    assert allowed is False and 0 < wait <= 10
    assert limiter.allow("b", "all", 2)[0] is True  # another client is unaffected
    assert limiter.allow("a", "heavy", 2)[0] is True  # another bucket is unaffected
    now[0] = 11.0
    assert limiter.allow("a", "all", 2)[0] is True  # the window slid past the old hits


def test_api_returns_429_with_retry_after() -> None:
    cfg = config_with(max_requests=3)
    with TestClient(create_app(make_state(), cfg)) as c:
        codes = [c.get("/search/text", params={"q": "shirt"}).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        r = c.get("/search/text", params={"q": "shirt"})
        assert r.json()["error"]["code"] == "rate_limited" and int(r.headers["Retry-After"]) >= 1
        assert c.get("/health").status_code == 200  # exempt paths are never limited


def test_llm_endpoints_have_a_stricter_limit() -> None:
    cfg = config_with(heavy_max_requests=2)
    with TestClient(create_app(make_state(), cfg)) as c:
        body = {"attributes": ATTRS}
        codes = [c.post("/generate-listing", json=body).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        assert c.get("/search/text", params={"q": "shirt"}).status_code == 200


def test_rate_limit_can_be_switched_off() -> None:
    with TestClient(create_app(make_state(), config_with(enabled=False, max_requests=1))) as c:
        assert all(
            c.get("/search/text", params={"q": "shirt"}).status_code == 200 for _ in range(5)
        )


def test_body_larger_than_the_cap_is_rejected_before_reading() -> None:
    with TestClient(create_app(make_state())) as c:
        huge = b"0" * (6 * 1024 * 1024)
        r = c.post("/predict-attributes", files=files(huge))
        assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"
        small = c.post("/search/image", files=files(png_bytes()))
        assert small.status_code == 200  # a normal photo is not affected


def test_bad_content_length_header_is_a_clear_400() -> None:
    with TestClient(create_app(make_state())) as c:
        r = c.post("/predict-attributes", content=b"x", headers={"content-length": "abc"})
        assert r.status_code == 400 and r.json()["error"]["code"] == "bad_content_length"
