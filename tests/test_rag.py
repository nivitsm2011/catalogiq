"""Phase 4 tests: BM25, fusion, categories, schema, guardrails, LLM wrapper, generator (mocked)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from catalogiq.llm.client import CachedClient, LLMClient, LLMError, LLMResponse
from catalogiq.rag.bm25 import BM25, tokenize
from catalogiq.rag.categories import group_from_article_type, group_from_title
from catalogiq.rag.guardrails import check_listing, mentions
from catalogiq.rag.listing import ListingGenerator, format_attributes
from catalogiq.rag.retrieval import HybridRetriever, rrf_fuse
from catalogiq.rag.schema import Listing

ATTRS = {
    "gender": ("Men", 0.95, False),
    "articleType": ("Tshirts", 0.9, False),
    "baseColour": ("Blue", 0.8, False),
    "usage": ("Casual", 0.4, True),
    "season": ("Summer", 0.7, False),
}


def good_listing(**over: Any) -> dict[str, Any]:
    base = {
        "title": "Puma Men's Short Sleeve T-Shirt, Blue",
        "bullets": [
            "Men's blue t-shirt",
            "Short sleeves",
            "Everyday style",
            "Summer ready",
            "Easy to pair",
        ],
        "description": " ".join(
            ["A simple blue t-shirt for men from Puma, suited to summer days."] * 4
        ),
        "search_tags": ["men tshirt", "blue tshirt", "puma", "summer tshirt", "mens tee"],
        "attributes_used": ["gender", "articleType", "baseColour"],
        "warnings": [],
    }
    base.update(over)
    return base


class FakeLLM(LLMClient):
    """Returns canned replies in order; records calls."""

    model = "fake"

    def __init__(self, replies: list[str]) -> None:
        self.replies, self.calls = list(replies), []

    def chat(self, messages, schema=None, temperature=None, max_tokens=None) -> LLMResponse:
        self.calls.append(messages)
        if not self.replies:
            raise LLMError("no more replies")
        return LLMResponse(self.replies.pop(0), 0.01)


# ----------------------------------------------------------------------------- retrieval pieces
def test_bm25_prefers_exact_rare_token() -> None:
    docs = ["blue cotton shirt for men", "sterling silver 925 earrings", "black leather wallet"]
    scores = BM25(docs).scores("925 earrings")
    assert int(np.argmax(scores)) == 1 and scores[0] == 0


def test_tokenize_drops_stopwords() -> None:
    assert tokenize("The T-Shirt for Men") == ["t", "shirt", "men"]


def test_rrf_rewards_agreement() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused["a"] == pytest.approx(fused["b"]) and fused["b"] > fused["c"] > fused["d"] - 1e-9
    assert set(fused) == {"a", "b", "c", "d"}


class FakeCollection:
    def __init__(self, ids: list[str], groups: list[str]) -> None:
        self.ids, self.groups = ids, groups

    def query(self, query_embeddings, n_results, where=None):
        pairs = zip(self.ids, self.groups, strict=True)
        keep = [i for i, g in pairs if not where or g == where["group"]]
        return {"ids": [keep[:n_results]]}


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parent_asin": ["a", "b", "c", "d"],
            "title": ["blue shirt", "blue tee", "red shirt", "silver ring"],
            "features": [[], [], [], []],
            "doc": ["blue shirt", "blue tee", "red shirt", "silver ring"],
            "group": ["topwear", "topwear", "topwear", "jewellery"],
            "quality_score": [0.5, 0.9, 0.5, 0.5],
        }
    )


def test_quality_boost_reorders_and_group_filter(monkeypatch) -> None:
    monkeypatch.setattr("catalogiq.rag.retrieval.embed_texts", lambda t: np.zeros((1, 3)))
    col = FakeCollection(["a", "b", "c", "d"], ["topwear", "topwear", "topwear", "jewellery"])
    r = HybridRetriever(_items(), collection=col, style_collection=None)
    plain = r.search("blue", k=3, mode="vector", boost=0.0)
    boosted = r.search("blue", k=3, mode="vector", boost=5.0)
    assert plain["parent_asin"].tolist()[0] == "a"
    assert boosted["parent_asin"].tolist()[0] == "b"
    ring = r.search("ring", k=2, group="jewellery", mode="vector")  # group too small -> filter off
    assert "a" in ring["parent_asin"].tolist()


# ----------------------------------------------------------------------------- categories
def test_group_rules() -> None:
    assert group_from_title("Men's Cotton Boxer Brief 6-pack") == "innerwear_socks"
    assert group_from_title("Casio Digital Watch") == "watches"
    assert group_from_title("Women's Maxi Dress with Pockets") == "dresses_ethnic"
    assert group_from_title("Dress Shirt for Men") == "topwear"
    assert group_from_article_type("Tshirts") == "topwear"
    assert group_from_article_type("Nonexistent") == "other"


# ----------------------------------------------------------------------------- schema
def test_schema_requires_five_bullets() -> None:
    Listing.model_validate(good_listing())
    with pytest.raises(ValidationError):
        Listing.model_validate(good_listing(bullets=["a", "b"]))


# ----------------------------------------------------------------------------- guardrails
def _check(listing: dict[str, Any], notes: dict[str, Any] | None = None, lex=None):
    return check_listing(
        Listing.model_validate(listing),
        ATTRS,
        notes or {"brand": "Puma"},
        ["gender", "articleType", "baseColour", "usage"],
        lex or ["nike", "puma"],
    )


def test_clean_listing_passes() -> None:
    rep = _check(good_listing())
    assert not rep.hallucinated and rep.style_compliant
    assert rep.missing_attributes == [] and rep.coverage == 1.0  # "Everyday" counts as casual usage


def test_unsupported_material_brand_claim_flagged() -> None:
    rep = _check(good_listing(bullets=["100% cotton", "Waterproof", "Nike style", "x", "y"]))
    kinds = {k for k, _ in rep.unsupported}
    assert {"material", "claim", "brand"} <= kinds


def test_material_allowed_when_in_seller_notes() -> None:
    rep = _check(
        good_listing(bullets=["Soft cotton", "a", "b", "c", "d"]),
        {"brand": "Puma", "material": "cotton"},
    )
    assert ("material", "cotton") not in rep.unsupported


def test_banned_and_title_rules() -> None:
    rep = _check(good_listing(title="Best T-Shirt", description="The perfect blue shirt " * 8))
    assert "best" in rep.banned and "perfect" in rep.banned
    assert any(v.startswith("title_length") for v in rep.style_violations)
    assert "brand_not_first" in rep.style_violations


def test_uncertain_attribute_in_title_flagged() -> None:
    rep = _check(good_listing(title="Puma Men's Casual T-Shirt, Blue"))
    assert "uncertain_usage_in_title" in rep.style_violations


def test_mentions() -> None:
    assert mentions("gender", "Men", "mens shirt")
    assert mentions("gender", "Men", "men's shirt")
    assert not mentions("gender", "Men", "women's shirt")
    assert mentions("articleType", "Tshirts", "a blue T-Shirt")
    assert mentions("baseColour", "Navy Blue", "navy shirt")


# ----------------------------------------------------------------------------- LLM wrapper
def test_cached_client_calls_inner_once(tmp_path) -> None:
    inner = FakeLLM(["one", "two"])
    cached = CachedClient(inner, tmp_path)
    msgs = [{"role": "user", "content": "hi"}]
    first, second = cached.chat(msgs), cached.chat(msgs)
    assert (first.text, second.text) == ("one", "one") and second.cached and len(inner.calls) == 1


# ----------------------------------------------------------------------------- generator
class FakeRetriever:
    def search(self, query, k, group=None, **_):
        return pd.DataFrame(
            {"parent_asin": ["x1"], "title": ["Example tee"], "features": [["Soft cotton fabric"]]}
        )

    def style_rules(self, group: str) -> str:
        return f"RULES for {group}"


def test_generate_no_rag_has_no_examples() -> None:
    llm = FakeLLM([json.dumps(good_listing())])
    res = ListingGenerator(llm, None, ["puma"]).generate(ATTRS, {"brand": "Puma"}, use_rag=False)
    prompt = llm.calls[0][1]["content"]
    assert (
        res.ok
        and res.attempts == 1
        and "EXAMPLE LISTINGS" not in prompt
        and "STYLE GUIDE" not in prompt
    )
    assert "UNCERTAIN" in prompt and any("usage" in w for w in res.listing.warnings)


def test_generate_rag_prompt_has_examples_and_style() -> None:
    llm = FakeLLM([json.dumps(good_listing())])
    res = ListingGenerator(llm, FakeRetriever(), []).generate(ATTRS, {}, use_rag=True)  # type: ignore[arg-type]
    prompt = llm.calls[0][1]["content"]
    assert (
        "EXAMPLE LISTINGS" in prompt and "RULES for topwear" in prompt and res.retrieved == ["x1"]
    )


def test_repair_retry_then_success_and_failure() -> None:
    llm = FakeLLM(["not json", json.dumps(good_listing())])
    res = ListingGenerator(llm, None).generate(ATTRS, use_rag=False)
    assert res.ok and res.attempts == 2 and "Problem" in llm.calls[1][-1]["content"]
    llm2 = FakeLLM(["bad", "still bad"])
    res2 = ListingGenerator(llm2, None).generate(ATTRS, use_rag=False)
    assert not res2.ok and res2.error and res2.attempts == 2


def test_llm_error_is_reported_not_raised() -> None:
    res = ListingGenerator(FakeLLM([]), None).generate(ATTRS, use_rag=False)
    assert not res.ok and "no more replies" in (res.error or "")


def test_format_attributes_marks_uncertain() -> None:
    text = format_attributes(ATTRS)
    assert "usage: Casual (UNCERTAIN" in text and "gender: Men (confident)" in text


def test_self_correct_keeps_only_better_rewrite() -> None:
    bad = good_listing(bullets=["100% cotton", "Waterproof", "b", "c", "d"])
    better = good_listing()
    llm = FakeLLM([json.dumps(bad), json.dumps(better)])
    res = ListingGenerator(llm, None, []).generate(ATTRS, {"brand": "Puma"}, False, True)
    assert res.fixed and res.fix_attempts == 1 and not res.report.hallucinated
    assert "unsupported" in llm.calls[1][-1]["content"]
    mild = good_listing(bullets=["Perfect blue tee", "a", "b", "c", "d"])
    worse = FakeLLM([json.dumps(mild), json.dumps(bad)])
    res2 = ListingGenerator(worse, None, []).generate(ATTRS, {"brand": "Puma"}, False, True)
    assert (
        not res2.fixed and res2.fix_attempts == 1 and res2.listing.bullets[0] == "Perfect blue tee"
    )


def test_generate_from_image_uses_classifier_and_passes_flags(monkeypatch) -> None:
    monkeypatch.setattr("catalogiq.vision.predict.predict", lambda image: ATTRS)
    llm = FakeLLM([json.dumps(good_listing())])
    res, attrs = ListingGenerator(llm, None, []).generate_from_image(
        "photo.jpg", {"brand": "Puma"}, use_rag=False, self_correct=True
    )
    assert attrs == ATTRS and res.ok
