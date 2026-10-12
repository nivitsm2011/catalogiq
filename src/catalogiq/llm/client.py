"""LLM wrapper: one small interface, Ollama by default, an on-disk cache, API models swap in here.

Everything else in the project talks to ``LLMClient.chat``; to use a hosted model later, add a class
that implements it and register it in ``PROVIDERS``.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catalogiq.config import get_config, get_rag_config, resolve_path
from catalogiq.logging_utils import get_logger

logger = get_logger(__name__)

Messages = list[dict[str, str]]


class LLMError(RuntimeError):
    """Raised when the model backend fails (not running, timeout, bad response)."""


@dataclass
class LLMResponse:
    """One model reply."""

    text: str
    latency_s: float
    cached: bool = False
    output_tokens: int | None = None


class LLMClient(ABC):
    """Interface every backend implements."""

    model: str

    @abstractmethod
    def chat(
        self,
        messages: Messages,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat; ``schema`` (JSON schema) asks for structured JSON output."""


class OllamaClient(LLMClient):
    """Local Ollama server.

    Args:
        model: Ollama model tag.
        host: Server URL.
        timeout_s: Per-request timeout in seconds.
        temperature: Default sampling temperature.
        seed: Sampling seed (makes runs repeatable).
    """

    def __init__(
        self, model: str, host: str, timeout_s: float, temperature: float, seed: int
    ) -> None:
        import ollama

        self.model = model
        self.temperature = temperature
        self.seed = seed
        self._client = ollama.Client(host=host, timeout=timeout_s)

    def chat(
        self,
        messages: Messages,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        options: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "seed": self.seed,
        }
        if max_tokens:
            options["num_predict"] = max_tokens
        start = time.perf_counter()
        try:
            res = self._client.chat(
                model=self.model, messages=messages, format=schema, options=options
            )
        except Exception as exc:  # noqa: BLE001 - connection, timeout and model errors alike
            raise LLMError(f"Ollama call failed: {exc}") from exc
        return LLMResponse(
            text=res.message.content or "",
            latency_s=time.perf_counter() - start,
            output_tokens=getattr(res, "eval_count", None),
        )


class CachedClient(LLMClient):
    """Wraps any client and stores replies on disk, keyed by model, messages, schema and settings.

    Identical requests are answered from the cache (``cached=True``); the original latency is kept
    in the entry so evaluation can still report real generation time.
    """

    def __init__(self, inner: LLMClient, cache_dir: str | Path) -> None:
        self.inner = inner
        self.model = inner.model
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, messages: Messages, schema: Any, temperature: Any, max_tokens: Any) -> str:
        blob = json.dumps(
            [self.model, messages, schema, temperature, max_tokens],
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(
        self,
        messages: Messages,
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        path = self.dir / f"{self._key(messages, schema, temperature, max_tokens)}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return LLMResponse(
                data["text"], data["latency_s"], cached=True, output_tokens=data.get("tokens")
            )
        res = self.inner.chat(messages, schema, temperature, max_tokens)
        path.write_text(
            json.dumps({"text": res.text, "latency_s": res.latency_s, "tokens": res.output_tokens}),
            encoding="utf-8",
        )
        return res


def _make_ollama() -> LLMClient:
    cfg = get_config()["models"]["llm"]
    return OllamaClient(
        model=cfg["model"],
        host=cfg["host"],
        timeout_s=cfg["timeout_s"],
        temperature=cfg["temperature"],
        seed=get_rag_config()["llm"]["seed"],
    )


PROVIDERS = {"ollama": _make_ollama}


def get_llm_client(cache: bool | None = None) -> LLMClient:
    """Build the configured client (``models.llm.provider``), disk-cached by default."""
    provider = get_config()["models"]["llm"]["provider"]
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider {provider!r}; known: {sorted(PROVIDERS)}")
    client = PROVIDERS[provider]()
    use_cache = get_rag_config()["llm"]["cache"] if cache is None else cache
    if use_cache:
        client = CachedClient(client, resolve_path(get_rag_config()["paths"]["llm_cache_dir"]))
    return client
