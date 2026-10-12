"""Post-generation checks: hallucination (unsupported facts), banned words/claims, style compliance.

The allowed source of facts is ONLY the input: predicted attributes plus seller notes.
Retrieved example listings never count as a source, so a material copied from an example is flagged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from catalogiq.config import get_rag_config
from catalogiq.rag.schema import Listing

Attributes = dict[str, tuple[str, float, bool]]  # attribute -> (label, confidence, needs_review)


@dataclass
class GuardrailReport:
    """Result of all checks for one listing."""

    unsupported: list[tuple[str, str]] = field(default_factory=list)  # (kind, term)
    banned: list[str] = field(default_factory=list)
    style_violations: list[str] = field(default_factory=list)
    coverage: float = 1.0
    missing_attributes: list[str] = field(default_factory=list)

    @property
    def hallucinated(self) -> bool:
        """True when the output states any material, brand, claim or number the input does not."""
        return bool(self.unsupported)

    @property
    def style_compliant(self) -> bool:
        """True when no style rule is broken and no banned word or claim is used."""
        return not self.style_violations and not self.banned

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for logging and DataFrames."""
        return {
            "hallucinated": self.hallucinated,
            "unsupported": [f"{k}:{t}" for k, t in self.unsupported],
            "banned": self.banned,
            "style_violations": self.style_violations,
            "style_compliant": self.style_compliant,
            "coverage": self.coverage,
            "missing_attributes": self.missing_attributes,
        }


def _norm(text: str) -> str:
    return text.lower().replace("’", "'").replace("‘", "'")


def output_text(listing: Listing) -> str:
    """All customer-facing text (warnings and attributes_used are internal and excluded)."""
    parts = [listing.title, *listing.bullets, listing.description, *listing.search_tags]
    return _norm(" \n ".join(parts))


def allowed_text(attributes: Attributes, seller_notes: dict[str, Any]) -> str:
    """Everything the writer may legitimately state: attribute labels and seller-note values."""
    parts = [label for label, _, _ in attributes.values()]
    parts += [
        f"{conf:.0%}" for _, conf, _ in attributes.values()
    ]  # confidences shown to the writer
    parts += [str(v) for v in seller_notes.values() if v not in (None, "")]
    return _norm(" ; ".join(parts))


def _has(pattern: str, text: str) -> re.Match[str] | None:
    return re.search(pattern, text, re.IGNORECASE)


def mention_patterns(attribute: str, label: str) -> list[str]:
    """Regexes that count as mentioning ``label`` of ``attribute``."""
    terms = get_rag_config()["attribute_terms"]
    if attribute == "gender":
        words = terms["gender"].get(label, [label.lower()])
        return [rf"\b{re.escape(w)}\b" for w in [label.lower(), *words]]
    if attribute == "usage":
        words = terms["usage"].get(label, [label.lower()])
        return [rf"\b{re.escape(w)}" for w in [label.lower(), *words]]
    if attribute == "articleType":
        extra = terms["type_terms"].get(label, [])
        last = label.lower().split()[-1]
        stem = last[:-1] if last.endswith("s") and len(last) > 3 else last
        return [rf"{re.escape(s)}" for s in {stem, *extra}]  # matched on squashed text
    if attribute == "baseColour":
        words = {label.lower(), *label.lower().replace("-", " ").split()}
        return [rf"\b{re.escape(w)}\b" for w in words]
    return [rf"\b{re.escape(label.lower())}\b"]


def mentions(attribute: str, label: str, text: str) -> bool:
    """Does ``text`` mention this attribute value?"""
    text = _norm(text)
    if attribute == "articleType":
        squashed = re.sub(r"[^a-z0-9]", "", text)
        return any(re.search(p, squashed) for p in mention_patterns(attribute, label))
    return any(re.search(p, text) for p in mention_patterns(attribute, label))


def attribute_coverage(
    listing: Listing, attributes: Attributes, group_required: list[str]
) -> tuple[float, list[str]]:
    """Share of the category's required attributes (that exist in the input) mentioned in the copy.

    Warnings count as text: an uncertain attribute phrased as "please confirm" is still mentioned.
    """
    text = output_text(listing) + " " + _norm(" ".join(listing.warnings))
    needed = [a for a in group_required if a in attributes]
    if not needed:
        return 1.0, []
    missing = [a for a in needed if not mentions(a, attributes[a][0], text)]
    return 1.0 - len(missing) / len(needed), missing


def check_hallucination(
    listing: Listing,
    attributes: Attributes,
    seller_notes: dict[str, Any],
    brand_lexicon: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Materials, brands, claims and numbers in the output that the inputs do not contain."""
    cfg = get_rag_config()["guardrails"]
    text = output_text(listing)
    allowed = allowed_text(attributes, seller_notes)
    flagged: list[tuple[str, str]] = []
    for m in cfg["materials"]:
        if _has(rf"\b{m}\b", text) and not _has(rf"\b{m}\b", allowed):
            flagged.append(("material", m))
    for c in cfg["claims"]:
        hit = _has(c, text)
        if hit and not _has(c, allowed):
            flagged.append(("claim", hit.group(0)))
    for n in re.finditer(cfg["numeric_pattern"], text):
        token = re.sub(r"\s+", "", n.group(0))
        if token not in re.sub(r"\s+", "", allowed):
            flagged.append(("number", n.group(0).strip()))
    for s in re.finditer(cfg["size_pattern"], text):
        if not _has(re.escape(s.group(0)), allowed):
            flagged.append(("size", s.group(0)))
    for b in brand_lexicon or []:
        if _has(rf"(?<![a-z0-9]){re.escape(b)}(?![a-z0-9])", text) and not _has(
            rf"(?<![a-z0-9]){re.escape(b)}(?![a-z0-9])", allowed
        ):
            flagged.append(("brand", b))
    return list(dict.fromkeys(flagged))


def check_banned(listing: Listing) -> list[str]:
    """Banned hype words and banned (health / unverifiable) claims used anywhere in the copy."""
    style = get_rag_config()["style"]
    text = output_text(listing)
    hits = []
    for p in [*style["banned_words"], *style["banned_claims"]]:
        m = _has(p, text)
        if m:
            hits.append(m.group(0))
    return list(dict.fromkeys(hits))


def check_style(
    listing: Listing,
    attributes: Attributes,
    seller_notes: dict[str, Any],
) -> list[str]:
    """Title length, bullet/description/tag limits, brand-first, required parts in the title."""
    s = get_rag_config()["style"]
    v: list[str] = []
    n = len(listing.title)
    if n < s["title_min_chars"] or n > s["title_max_chars"]:
        v.append(f"title_length:{n}")
    if any(len(b) > s["bullet_max_chars"] for b in listing.bullets):
        v.append("bullet_too_long")
    words = len(listing.description.split())
    if words < s["description_min_words"] or words > s["description_max_words"]:
        v.append(f"description_words:{words}")
    if not s["tags_min"] <= len(listing.search_tags) <= s["tags_max"]:
        v.append(f"tag_count:{len(listing.search_tags)}")
    title = _norm(listing.title)
    brand = str(seller_notes.get("brand") or "").strip().lower()
    if brand and not title.startswith(brand):
        v.append("brand_not_first")
    positions: list[int] = []
    for attr in ("gender", "articleType", "baseColour"):
        if attr not in attributes or attributes[attr][2]:
            continue  # unknown or uncertain: not required in the title
        label = attributes[attr][0]
        if not mentions(attr, label, title):
            v.append(f"title_missing_{attr}")
            continue
        if attr != "articleType":
            m = next(
                (re.search(p, title) for p in mention_patterns(attr, label) if re.search(p, title)),
                None,
            )
            positions.append(m.start() if m else -1)
    for attr, (label, _, review) in attributes.items():
        if review and attr in ("gender", "articleType", "baseColour", "usage"):
            if any(re.search(p, title) for p in mention_patterns(attr, label)):
                v.append(f"uncertain_{attr}_in_title")
    if positions != sorted(positions):
        v.append("title_order")
    return v


def check_listing(
    listing: Listing,
    attributes: Attributes,
    seller_notes: dict[str, Any],
    required_attributes: list[str],
    brand_lexicon: list[str] | None = None,
) -> GuardrailReport:
    """Run every guardrail on one generated listing."""
    coverage, missing = attribute_coverage(listing, attributes, required_attributes)
    return GuardrailReport(
        unsupported=check_hallucination(listing, attributes, seller_notes, brand_lexicon),
        banned=check_banned(listing),
        style_violations=check_style(listing, attributes, seller_notes),
        coverage=coverage,
        missing_attributes=missing,
    )
