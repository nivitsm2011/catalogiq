"""LLM-as-judge: 1-5 clarity and accuracy scores (prompt: ``configs/prompts/judge.yaml``)."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from catalogiq.config import get_rag_config, resolve_path
from catalogiq.llm.client import LLMClient, LLMError
from catalogiq.rag.guardrails import Attributes
from catalogiq.rag.listing import format_attributes, format_seller_notes
from catalogiq.rag.schema import Listing


class JudgeScore(BaseModel):
    """Judge output."""

    clarity: int = Field(ge=1, le=5)
    accuracy: int = Field(ge=1, le=5)
    reason: str = ""


@lru_cache(maxsize=1)
def _prompts() -> dict[str, str]:
    path = resolve_path(get_rag_config()["paths"]["prompts_judge"])
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def judge_listing(
    llm: LLMClient,
    listing: Listing,
    attributes: Attributes,
    seller_notes: dict[str, Any],
) -> JudgeScore | None:
    """Score one listing; ``None`` when the judge fails to return valid JSON twice."""
    p = _prompts()
    user = p["user"].format(
        attributes=format_attributes(attributes),
        seller_notes=format_seller_notes(seller_notes),
        title=listing.title,
        bullets="\n".join(f"- {b}" for b in listing.bullets),
        description=listing.description,
        tags=", ".join(listing.search_tags),
    )
    messages = [{"role": "system", "content": p["system"]}, {"role": "user", "content": user}]
    temp = get_rag_config()["llm"]["judge_temperature"]
    for _ in range(2):
        try:
            reply = llm.chat(messages, schema=JudgeScore.model_json_schema(), temperature=temp)
            return JudgeScore.model_validate(json.loads(reply.text))
        except (LLMError, ValidationError, json.JSONDecodeError):
            continue
    return None
