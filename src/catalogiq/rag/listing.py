"""Grounded listing generator: image -> attributes -> retrieval -> LLM -> JSON -> guardrails."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import yaml
from pydantic import ValidationError

from catalogiq.config import get_rag_config, resolve_path
from catalogiq.llm.client import LLMClient, LLMError
from catalogiq.logging_utils import get_logger
from catalogiq.rag.categories import group_from_article_type, group_spec
from catalogiq.rag.guardrails import Attributes, GuardrailReport, check_listing
from catalogiq.rag.retrieval import HybridRetriever
from catalogiq.rag.schema import Listing, listing_json_schema

logger = get_logger(__name__)
NL = chr(10)


@lru_cache(maxsize=1)
def load_prompts() -> dict[str, str]:
    """Prompt templates from ``configs/prompts/listing.yaml``."""
    path = resolve_path(get_rag_config()["paths"]["prompts_listing"])
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass
class GenerationResult:
    """Everything produced for one product."""

    listing: Listing | None
    report: GuardrailReport | None
    group: str
    use_rag: bool
    latency_s: float = 0.0
    attempts: int = 0
    retrieved: list[str] = field(default_factory=list)  # asins of the examples shown
    error: str | None = None
    raw: list[str] = field(default_factory=list)  # raw model replies (1 or 2)
    messages: list[dict[str, str]] = field(default_factory=list)
    fix_attempts: int = 0  # guardrail-driven rewrites tried (0 or 1)
    fixed: bool = False  # True when the rewrite replaced the first draft

    @property
    def ok(self) -> bool:
        """True when valid JSON was produced."""
        return self.listing is not None


def format_attributes(attributes: Attributes) -> str:
    """Attribute lines for the prompt; low-confidence ones are marked UNCERTAIN."""
    keys = get_rag_config()["generation"]["prompt_attributes"]
    lines = []
    for key in keys:
        if key not in attributes:
            continue
        label, conf, review = attributes[key]
        if review:
            lines.append(f"- {key}: {label} (UNCERTAIN, confidence {conf:.0%}; please confirm)")
        else:
            lines.append(f"- {key}: {label} (confident)")
    return "\n".join(lines) or "- (none)"


def format_seller_notes(seller_notes: dict[str, Any]) -> str:
    """Seller-note lines for the prompt."""
    lines = [f"- {k}: {v}" for k, v in seller_notes.items() if v not in (None, "")]
    return "\n".join(lines) or "- (none provided)"


def retrieval_query(attributes: Attributes, seller_notes: dict[str, Any]) -> str:
    """Free-text query built from the confident-or-not attribute labels and seller notes."""
    parts = [seller_notes.get("brand"), seller_notes.get("material")]
    for key in ("gender", "baseColour", "articleType", "usage"):
        if key in attributes:
            parts.append(attributes[key][0])
    return " ".join(str(p) for p in parts if p)


class ListingGenerator:
    """Writes grounded listing copy.

    Args:
        llm: Any ``LLMClient`` (Ollama by default; a fake in tests).
        retriever: Hybrid retriever over the knowledge base; required for ``use_rag=True``.
        brand_lexicon: Known brand names for the hallucination check.
    """

    def __init__(
        self,
        llm: LLMClient,
        retriever: HybridRetriever | None = None,
        brand_lexicon: list[str] | None = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.brand_lexicon = brand_lexicon or []
        self.cfg = get_rag_config()
        self.prompts = load_prompts()

    # ------------------------------------------------------------------ prompt
    def build_messages(
        self, attributes: Attributes, seller_notes: dict[str, Any], use_rag: bool
    ) -> tuple[list[dict[str, str]], str, list[str]]:
        """Chat messages, category group and the asins of the retrieved examples."""
        article = attributes.get("articleType", ("", 0.0, True))[0]
        group = group_from_article_type(article)
        style_section = examples_section = ""
        retrieved: list[str] = []
        if use_rag:
            if self.retriever is None:
                raise ValueError("use_rag=True needs a retriever")
            s, r = self.cfg["style"], self.cfg["retrieval"]
            style_section = self.prompts["style_section"].format(
                style_rules=self.retriever.style_rules(group),
                title_template=s["title_template"],
                title_example=s["title_example"],
                title_min=s["title_min_chars"],
                title_max=s["title_max_chars"],
                desc_min=s["description_min_words"],
                desc_max=s["description_max_words"],
                tags_min=s["tags_min"],
                tags_max=s["tags_max"],
                bullet_max=s["bullet_max_chars"],
            )
            hits = self.retriever.search(
                retrieval_query(attributes, seller_notes), k=r["top_k"], group=group
            )
            blocks = []
            for n, row in enumerate(hits.itertuples(), start=1):
                feats = [
                    str(f)[: r["example_max_chars"]]
                    for f in (row.features if row.features is not None else [])
                ]
                feats = feats[: r["example_max_features"]]
                blocks.append(
                    self.prompts["example_item"].format(
                        n=n,
                        title=row.title[: r["example_max_chars"]],
                        features="\n".join(f"  - {f}" for f in feats) or "  (no bullet points)",
                    )
                )
                retrieved.append(row.parent_asin)
            examples_section = self.prompts["examples_section"].format(examples="\n".join(blocks))
        user = self.prompts["user"].format(
            attributes=format_attributes(attributes),
            seller_notes=format_seller_notes(seller_notes),
            style_section=style_section,
            examples_section=examples_section,
        )
        messages = [
            {"role": "system", "content": self.prompts["system"]},
            {"role": "user", "content": user},
        ]
        return messages, group, retrieved

    # --------------------------------------------------------------- generation
    @staticmethod
    def _parse(text: str) -> Listing:
        return Listing.model_validate(json.loads(text))

    def generate(
        self,
        attributes: Attributes,
        seller_notes: dict[str, Any] | None = None,
        use_rag: bool = True,
        self_correct: bool = False,
    ) -> GenerationResult:
        """Generate, validate (one repair retry), then run the guardrails.

        With ``self_correct`` the guardrail findings are sent back to the model once and the
        rewrite is kept only if it has fewer findings than the first draft.
        """
        seller_notes = seller_notes or {}
        messages, group, retrieved = self.build_messages(attributes, seller_notes, use_rag)
        result = GenerationResult(
            None, None, group, use_rag, retrieved=retrieved, messages=messages
        )
        schema = listing_json_schema()
        llm_cfg = self.cfg["llm"]
        convo = list(messages)
        for attempt in range(1 + llm_cfg["repair_attempts"]):
            result.attempts = attempt + 1
            try:
                reply = self.llm.chat(convo, schema=schema, max_tokens=llm_cfg["max_new_tokens"])
            except LLMError as exc:
                result.error = str(exc)
                logger.error("LLM failure: %s", exc)
                return result
            result.latency_s += reply.latency_s
            result.raw.append(reply.text)
            try:
                result.listing = self._parse(reply.text)
                result.error = None
                break
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                result.error = str(exc)[: self.cfg["generation"]["max_title_retry_text"]]
                convo = [
                    *messages,
                    {"role": "assistant", "content": reply.text},
                    {"role": "user", "content": self.prompts["repair"].format(error=result.error)},
                ]
        if result.listing is None:
            return result
        self._add_confirm_warnings(result.listing, attributes)
        result.report = check_listing(
            result.listing,
            attributes,
            seller_notes,
            group_spec(group)["required_attributes"],
            self.brand_lexicon,
        )
        if self_correct and self._n_findings(result.report):
            self._self_correct(result, messages, attributes, seller_notes)
        return result

    @staticmethod
    def _n_findings(report: GuardrailReport) -> int:
        return len(report.unsupported) + len(report.banned) + len(report.style_violations)

    def _self_correct(
        self,
        result: GenerationResult,
        messages: list[dict[str, str]],
        attributes: Attributes,
        seller_notes: dict[str, Any],
    ) -> None:
        """One guardrail-driven rewrite; the result is replaced only if it is strictly better."""
        report = result.report
        assert report is not None and result.listing is not None
        problems = [f"unsupported {k}: '{t}'" for k, t in report.unsupported]
        problems += [f"banned word: '{b}'" for b in report.banned]
        problems += [f"style: {v}" for v in report.style_violations]
        convo = [
            *messages,
            {"role": "assistant", "content": result.raw[-1]},
            {"role": "user", "content": self.prompts["fix"].format(problems=NL.join(problems))},
        ]
        try:
            reply = self.llm.chat(
                convo, schema=listing_json_schema(), max_tokens=self.cfg["llm"]["max_new_tokens"]
            )
            result.latency_s += reply.latency_s
            result.fix_attempts = 1
            new = self._parse(reply.text)
        except (LLMError, ValidationError, json.JSONDecodeError, ValueError):
            return
        self._add_confirm_warnings(new, attributes)
        new_report = check_listing(
            new,
            attributes,
            seller_notes,
            group_spec(result.group)["required_attributes"],
            self.brand_lexicon,
        )
        if self._n_findings(new_report) < self._n_findings(report):
            result.listing, result.report = new, new_report
            result.raw.append(reply.text)
            result.fixed = True

    @staticmethod
    def _add_confirm_warnings(listing: Listing, attributes: Attributes) -> None:
        """Make sure every uncertain attribute produces a 'please confirm' warning."""
        text = " ".join(listing.warnings).lower()
        for key, (label, conf, review) in attributes.items():
            if review and key in ("gender", "articleType", "baseColour", "usage", "season"):
                if label.lower() not in text:
                    listing.warnings.append(
                        f"Please confirm {key}: predicted '{label}' ({conf:.0%} confidence)."
                    )

    def generate_from_image(
        self,
        image: Any,
        seller_notes: dict[str, Any] | None = None,
        use_rag: bool = True,
        self_correct: bool = False,
    ) -> tuple[GenerationResult, Attributes]:
        """Photo -> Phase 2 attributes -> listing; returns the result and the attributes."""
        from catalogiq.vision.predict import predict

        attributes = predict(image)
        return self.generate(attributes, seller_notes, use_rag, self_correct), attributes
