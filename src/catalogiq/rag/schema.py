"""Validated output of the listing generator."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class Listing(BaseModel):
    """Generated listing copy. Structure is enforced here; style rules are checked by guardrails."""

    title: str = Field(min_length=1)
    bullets: list[str]
    description: str = Field(min_length=1)
    search_tags: list[str] = Field(min_length=1)
    attributes_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("bullets")
    @classmethod
    def _five_bullets(cls, v: list[str]) -> list[str]:
        v = [b.strip() for b in v if b and b.strip()]
        if len(v) != 5:
            raise ValueError(f"expected exactly 5 bullets, got {len(v)}")
        return v

    @field_validator("title", "description")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @field_validator("search_tags")
    @classmethod
    def _clean_tags(cls, v: list[str]) -> list[str]:
        seen: list[str] = []
        for t in v:
            t = t.strip().lower()
            if t and t not in seen:
                seen.append(t)
        if not seen:
            raise ValueError("search_tags is empty")
        return seen


def listing_json_schema() -> dict[str, Any]:
    """JSON schema passed to the model for structured output."""
    return Listing.model_json_schema()
