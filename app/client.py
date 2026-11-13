"""Thin client for the CatalogIQ API used by every Streamlit page."""

from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st

BASE_URL = os.environ.get("CATALOGIQ_API_URL", "http://127.0.0.1:8000").rstrip("/")
QUICK, SLOW = 30, 400  # seconds: normal calls, and calls that may wait for the local LLM


class ApiUnavailable(Exception):
    """The API cannot be reached."""


class ApiError(Exception):
    """The API answered with an error; ``message`` is safe to show to the user."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def url(path: str) -> str:
    """Absolute URL of an API path (for images the browser loads directly)."""
    return path if path.startswith("http") else f"{BASE_URL}{path}"


def _call(method: str, path: str, timeout: int = QUICK, **kwargs: Any) -> requests.Response:
    try:
        resp = requests.request(method, url(path), timeout=timeout, **kwargs)
    except requests.ConnectionError as exc:
        raise ApiUnavailable(
            f"The CatalogIQ API is not running at {BASE_URL}. Start it with: "
            "python -m uvicorn api.main:app --port 8000"
        ) from exc
    except requests.Timeout as exc:
        raise ApiError(
            504, "timeout", "The request took too long (the local model is slow on CPU)."
        ) from exc
    if resp.status_code >= 400:
        try:
            err = resp.json()["error"]
            raise ApiError(
                resp.status_code, err.get("code", "error"), err.get("message", resp.text)
            )
        except (ValueError, KeyError):
            raise ApiError(resp.status_code, "error", resp.text[:300]) from None
    return resp


def get_json(path: str, timeout: int = QUICK, **params: Any) -> Any:
    """GET a JSON endpoint."""
    return _call(
        "GET", path, timeout, params={k: v for k, v in params.items() if v not in (None, "")}
    ).json()


def post_json(path: str, body: dict[str, Any], timeout: int = QUICK) -> Any:
    """POST a JSON body."""
    return _call("POST", path, timeout, json=body).json()


def post_photo(
    path: str,
    photo: bytes,
    name: str = "photo.jpg",
    ctype: str = "image/jpeg",
    data: dict[str, Any] | None = None,
    timeout: int = QUICK,
) -> Any:
    """POST a photo (multipart) with optional form fields."""
    form = {k: str(v) for k, v in (data or {}).items() if v not in (None, "")}
    return _call("POST", path, timeout, files={"file": (name, photo, ctype)}, data=form).json()


def post_form(path: str, data: dict[str, Any], timeout: int = QUICK) -> Any:
    """POST form fields without a file."""
    return _call("POST", path, timeout, data={k: str(v) for k, v in data.items()}).json()


# ------------------------------------------------------------------------- cached reads
@st.cache_data(ttl=20, show_spinner=False)
def health() -> dict[str, Any]:
    """API health (cached for a few seconds)."""
    return get_json("/health", 5)


@st.cache_data(ttl=600, show_spinner=False)
def options() -> dict[str, Any]:
    """Dropdown values for attributes and search filters."""
    return get_json("/meta/options")


@st.cache_data(ttl=600, show_spinner=False)
def demo_products() -> list[dict[str, Any]]:
    """The 10 preloaded demo products."""
    return get_json("/demo/products")


@st.cache_data(ttl=600, show_spinner=False)
def image_bytes(path: str) -> bytes:
    """A catalog photo (cached)."""
    return _call("GET", path).content


@st.cache_data(ttl=600, show_spinner=False)
def review_products() -> list[dict[str, Any]]:
    """Products with reviews."""
    return get_json("/reviews/products", 60)


@st.cache_data(ttl=600, show_spinner=False)
def review_summary(item_id: str, phrase: bool = False) -> dict[str, Any]:
    """Aspect summary of a product's reviews."""
    return get_json(
        f"/reviews/{item_id}/summary", SLOW if phrase else 60, phrase=str(phrase).lower()
    )


@st.cache_data(ttl=120, show_spinner=False)
def ops_summary(category: str | None = None, limit: int = 12) -> dict[str, Any]:
    """Quality KPIs and worst listings."""
    return get_json("/ops/summary", 120, category=category, limit=limit)


@st.cache_data(ttl=60, show_spinner=False)
def ops_duplicates(limit: int = 6, version: int = 0) -> dict[str, Any]:
    """The duplicate review queue (``version`` busts the cache after a decision)."""
    del version
    return get_json("/ops/duplicates", 300, limit=limit)


@st.cache_data(ttl=300, show_spinner=False)
def scorecard() -> str:
    """The evaluation scorecard (markdown)."""
    return _call("GET", "/scorecard").text


@st.cache_data(ttl=300, show_spinner=False)
def search_text(q: str, k: int, filters_json: str) -> dict[str, Any]:
    """Text search with filters (filters passed as JSON so the call can be cached)."""
    return get_json("/search/text", 60, q=q, k=k, **json.loads(filters_json))


@st.cache_data(ttl=300, show_spinner=False)
def scorecard_metrics() -> dict[str, Any]:
    """Headline metrics of the last evaluation run."""
    return get_json("/scorecard/metrics")
