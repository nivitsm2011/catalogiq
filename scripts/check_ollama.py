"""One-line Ollama test prompt using the model configured in configs/base.yaml."""

from __future__ import annotations

import ollama

from catalogiq.config import get_config
from catalogiq.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Send a short prompt to the local LLM and log the reply."""
    llm = get_config()["models"]["llm"]
    client = ollama.Client(host=llm["host"], timeout=llm["timeout_s"])
    reply = client.chat(
        model=llm["model"],
        messages=[{"role": "user", "content": "Reply with exactly: CatalogIQ ready"}],
        options={"temperature": llm["temperature"]},
    )
    logger.info("model=%s reply=%s", llm["model"], reply["message"]["content"].strip())


if __name__ == "__main__":
    main()
