"""Category style guide: rendered from ``configs/rag.yaml`` so docs and checks never drift.

``docs/style_guide.md`` is the human-readable guide; the same rules are split into one retrievable
chunk per category (plus one global chunk) and indexed in the vector store.
"""

from __future__ import annotations

from typing import Any

from catalogiq.config import get_rag_config, resolve_path
from catalogiq.logging_utils import get_logger

logger = get_logger(__name__)

GLOBAL_GROUP = "all"


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


def global_rules(cfg: dict[str, Any] | None = None) -> str:
    """Rules that apply to every category, as plain text."""
    s = (cfg or get_rag_config())["style"]
    return (
        f"Title format: {s['title_template']}. Example: {s['title_example']}. "
        f"Title length {s['title_min_chars']} to {s['title_max_chars']} characters. "
        f"Exactly {s['bullets']} bullets, each at most {s['bullet_max_chars']} characters. "
        f"Description {s['description_min_words']} to {s['description_max_words']} words. "
        f"{s['tags_min']} to {s['tags_max']} search tags. "
        "Banned: superlatives and hype (best, perfect, #1, guaranteed, amazing, premium quality), "
        "sales language (sale, free shipping, buy now), and health or unverifiable benefit claims "
        "(cures, heals, relieves pain, hypoallergenic, anti-bacterial, clinically proven). "
        "Never state a material, brand, size, pack size or performance claim "
        "that the seller did not give. "
        "Attributes marked uncertain must be phrased as 'please confirm'."
    )


def group_rules(group: str, cfg: dict[str, Any] | None = None) -> str:
    """Category-specific rules (required attributes and guidance) as plain text."""
    cfg = cfg or get_rag_config()
    spec = cfg["categories"]["groups"][group]
    req = ", ".join(spec["required_attributes"])
    return (
        f"Category {group}. Required attributes to mention: {req}. "
        f"Title key attribute: usage or style when confident. {spec['guidance']}"
    )


def style_chunks(cfg: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """One chunk per category plus the global chunk: ``[{id, group, text}]``."""
    cfg = cfg or get_rag_config()
    chunks = [{"id": "style_all", "group": GLOBAL_GROUP, "text": global_rules(cfg)}]
    for group in cfg["categories"]["groups"]:
        chunks.append({"id": f"style_{group}", "group": group, "text": group_rules(group, cfg)})
    return chunks


def render_markdown(cfg: dict[str, Any] | None = None) -> str:
    """Full ``docs/style_guide.md`` text."""
    cfg = cfg or get_rag_config()
    s = cfg["style"]
    groups = cfg["categories"]["groups"]
    lines = [
        "# CatalogIQ category style guide",
        "",
        "_Generated from `configs/rag.yaml` by `scripts/build_knowledge_base.py`. Edit the config, "
        "not this file._",
        "",
        "These rules are (1) indexed in the vector store so the generator can retrieve "
        "the ones for "
        "a product's category and (2) enforced after generation by `catalogiq.rag.guardrails`.",
        "",
        "## Rules for every category",
        "",
        "### Title",
        f"- Format: `{s['title_template']}` (skip a part when that fact is unknown).",
        f"- Example: `{s['title_example']}`.",
        f"- Length: {s['title_min_chars']} to {s['title_max_chars']} characters.",
        "- Start with the brand when the seller gave one. Never invent a brand.",
        "- Only confident attributes go in the title. Uncertain ones are left out.",
        "",
        "### Bullets, description, tags",
        f"- Exactly {s['bullets']} bullets, each at most {s['bullet_max_chars']} characters, "
        "one fact per bullet.",
        f"- Description: {s['description_min_words']} to {s['description_max_words']} words, "
        "plain sentences, no hype.",
        f"- {s['tags_min']} to {s['tags_max']} search tags, lower case, buyer search words "
        "(type, gender, colour, occasion).",
        "",
        "### Banned words (hype and sales language)",
        _bullets([f"`{w}`" for w in s["banned_words"]]),
        "",
        "### Banned claims (health or unverifiable benefits)",
        _bullets([f"`{w}`" for w in s["banned_claims"]]),
        "",
        "### Facts policy",
        "- Use only facts from the predicted attributes and the seller notes.",
        "- Materials, brands, sizes, pack sizes and performance claims (waterproof, breathable, "
        "machine washable, UV protection ...) need to be in the seller notes.",
        "- An attribute the classifier is not sure about is written as _please confirm_ "
        "and listed in `warnings`.",
        "",
        "## Rules per category",
        "",
        "| Category | Required attributes | Example Kaggle article types |",
        "|---|---|---|",
    ]
    for name, spec in groups.items():
        lines.append(
            f"| {name} | {', '.join(spec['required_attributes'])} | "
            f"{', '.join(spec['article_types'][:4]) or '-'} |"
        )
    lines.append("")
    for name, spec in groups.items():
        lines += [
            f"### {name}",
            f"- Required: {', '.join(spec['required_attributes'])}.",
            f"- {spec['guidance']}",
            "",
        ]
    return "\n".join(lines)


def write_style_guide() -> str:
    """Write ``docs/style_guide.md`` from the config and return its path."""
    path = resolve_path(get_rag_config()["paths"]["style_guide_md"])
    path.write_text(render_markdown(), encoding="utf-8")
    logger.info("Wrote %s", path)
    return str(path)
