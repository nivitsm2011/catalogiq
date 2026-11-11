"""Phase 5 tests: chunking, aspect rules, summaries, Q&A pipeline (LLM and vector store mocked)."""

from __future__ import annotations

import json
import zlib
from typing import Any

import numpy as np
import pandas as pd
import pytest

from catalogiq.llm.client import LLMClient, LLMError, LLMResponse
from catalogiq.rag.aspects import best_threshold, prf, rule_matches, rule_tag
from catalogiq.rag.eval_qa import mean_ci, recall_at_k, validate_questions
from catalogiq.rag.qa import QAAnswer, ReviewQA, question_aspect
from catalogiq.rag.reviews_index import build_chunks, chunk_review
from catalogiq.rag.summary import (
    fit_direction,
    numbers_in,
    phrase_summary,
    product_facts,
    strip_label,
    tag_review_aspects,
    template_summary,
)

DIM = 32


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic bag-of-words hashing embedding (L2-normalised)."""
    out = np.zeros((len(texts), DIM), dtype="float32")
    for i, t in enumerate(texts):
        for w in t.lower().replace(".", " ").split():
            out[i, zlib.crc32(w.encode()) % DIM] += 1.0
        n = np.linalg.norm(out[i])
        out[i] /= n or 1.0
    return out


class FakeLLM(LLMClient):
    model = "fake"

    def __init__(self, replies: list[str]) -> None:
        self.replies, self.calls = list(replies), []

    def chat(self, messages, schema=None, temperature=None, max_tokens=None) -> LLMResponse:
        self.calls.append(messages)
        if not self.replies:
            raise LLMError("no more replies")
        return LLMResponse(self.replies.pop(0), 0.01)


class FakeCollection:
    """Stands in for a Chroma collection over the chunk table."""

    def __init__(self, chunks: pd.DataFrame) -> None:
        self.chunks = chunks.reset_index(drop=True)
        self.emb = fake_embed(self.chunks["text"].tolist())

    def query(self, query_embeddings, n_results, where=None):
        mask = (self.chunks["item_id"] == where["item_id"]).to_numpy()
        sims = self.emb @ np.asarray(query_embeddings[0])
        sims = np.where(mask, sims, -9)
        order = np.argsort(-sims)[:n_results]
        return {"ids": [[self.chunks["chunk_id"].iloc[i] for i in order if mask[i]]]}

    def get(self, ids, include=None):
        idx = {c: i for i, c in enumerate(self.chunks["chunk_id"])}
        return {"ids": list(ids), "embeddings": [self.emb[idx[i]] for i in ids]}


def make_chunks() -> pd.DataFrame:
    texts = [
        ("R1", "Runs small, order a size up. The fit is tight in the waist.", 5, True, 12),
        ("R2", "Too small for me, the size runs small and tight.", 2, True, 3),
        ("R3", "True to size and fits perfectly.", 5, False, 0),
        ("R4", "Shipping was fast and the box arrived on time.", 5, True, 1),
        ("R5", "The colour is a bit darker than the picture.", 4, True, 0),
    ]
    rows = [
        {
            "chunk_id": f"{rid}_0",
            "review_id": rid,
            "item_id": "A1",
            "rating": float(r),
            "verified_purchase": v,
            "helpful_vote": h,
            "date": "2021-01-01",
            "chunk_idx": 0,
            "n_chunks": 1,
            "text": t,
        }
        for rid, t, r, v, h in texts
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def qa(monkeypatch):
    monkeypatch.setattr("catalogiq.rag.qa.embed_texts", fake_embed)
    chunks = make_chunks()

    def build(replies: list[str], min_similarity: float = 0.2) -> tuple[ReviewQA, FakeLLM]:
        llm = FakeLLM(replies)
        return (
            ReviewQA(llm, chunks, FakeCollection(chunks), {"A1": "Test dress"}, min_similarity),
            llm,
        )

    return build


# ------------------------------------------------------------------ chunking
def test_short_review_kept_whole_with_title() -> None:
    out = chunk_review("Great fit", "It fits well.", 400, 3, 2, 700, True)
    assert out == ["Great fit. It fits well."]


def test_long_review_split_into_overlapping_windows() -> None:
    text = " ".join(f"Sentence number {i} is here." for i in range(8))
    out = chunk_review("", text, 20, 3, 2, 700, False)
    assert len(out) >= 3 and all(len(c) <= 700 for c in out)
    assert (
        "Sentence number 2" in out[0] and "Sentence number 2" in out[1]
    )  # overlap of one sentence


def test_build_chunks_keeps_metadata() -> None:
    rv = pd.DataFrame(
        {
            "parent_asin": ["A1"],
            "rating": [4.0],
            "title": ["t"],
            "text": ["Nice shirt, fits well."],
            "helpful_vote": [3],
            "verified_purchase": [True],
            "timestamp": [pd.Timestamp("2021-05-02")],
        }
    )
    c = build_chunks(rv).iloc[0]
    assert (c.review_id, c.item_id, c.helpful_vote, c.date) == ("R0", "A1", 3, "2021-05-02")


# ------------------------------------------------------------------ aspects
def test_rule_tagging_and_question_aspect() -> None:
    tags = rule_tag(["Runs small, order a size up", "Shipping was fast", "Hello"])
    assert tags[0].tolist() == [True, False, False, False, False, False]
    assert tags[1][5] and not tags[2].any()
    assert rule_matches("true to size")["fit_size"] == ["true to size"]
    assert question_aspect("Is the colour as pictured?") == "colour_accuracy"
    assert question_aspect("Does it come in blue?") in {"colour_accuracy", "general"}
    assert question_aspect("Hello there") == "general"


def test_prf_and_threshold() -> None:
    y = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=bool)
    p = y.copy()
    p[1, 0] = False
    res = prf(y, p).set_index("aspect")
    assert res.loc["fit_size", "precision"] == 1.0 and res.loc["fit_size", "recall"] == 0.5
    scores = np.array([[0.9, 0, 0, 0, 0, 0], [0.8, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0]])
    assert best_threshold(scores, y, [0.05, 0.5, 0.95]) == 0.5


# ------------------------------------------------------------------ summaries
def test_fit_direction_counts() -> None:
    d = fit_direction(["It runs small", "runs large and loose", "true to size", "nothing about it"])
    assert (d["runs_small"], d["runs_large"], d["true_to_size"], d["with_direction"]) == (
        1,
        1,
        1,
        3,
    )
    assert d["pct_runs_small"] == 33


def test_product_facts_and_template() -> None:
    chunks = make_chunks()
    ra = tag_review_aspects(chunks)
    facts = product_facts("A1", ra, dict(zip(chunks["review_id"], chunks["text"], strict=True)))
    # only 5 reviews: below min_mentions (5) for every aspect
    assert facts["aspects"] == {}
    assert "Not enough" in template_summary(facts)


def _facts() -> dict[str, Any]:
    return {
        "item_id": "A1",
        "n_reviews": 40,
        "aspects": {
            "fit_size": {
                "mentions": 20,
                "pct_of_reviews": 50,
                "pct_positive": 40,
                "pct_negative": 45,
                "pct_mixed": 15,
                "mean_rating": 3.1,
                "fit": {
                    "runs_small": 8,
                    "runs_large": 2,
                    "true_to_size": 2,
                    "with_direction": 12,
                    "pct_runs_small": 67,
                    "pct_runs_large": 17,
                    "pct_true_to_size": 17,
                },
            }
        },
    }


def test_template_uses_only_computed_numbers() -> None:
    text = template_summary(_facts())
    assert "20 reviews" in text and "67% say it runs small" in text
    assert numbers_in(text) <= numbers_in(json.dumps(_facts()))


def test_phrase_summary_checks_numbers_per_aspect() -> None:
    good = FakeLLM([json.dumps({"fit_size": "20 reviews mention fit and 67% say it runs small."})])
    res = phrase_summary(_facts(), "Dress", good)
    assert res["source"] == "llm" and res["bad_aspects"] == []
    assert res["summary"].startswith("Fit and size: 20 reviews")
    wrong = FakeLLM([json.dumps({"fit_size": "Fit: 99% say it runs small."})])
    res = phrase_summary(_facts(), "Dress", wrong)
    assert res["source"] == "template" and res["bad_numbers"] == ["99"]
    assert "67% say it runs small" in res["summary"]
    assert phrase_summary(_facts(), "Dress", FakeLLM([]))["source"] == "template"


def test_repeated_label_is_not_doubled() -> None:
    assert strip_label("Comfort: 9 reviews mention it.") == "9 reviews mention it."
    assert strip_label("Quality/Durability: 76 reviews, 88% positive") == "76 reviews, 88% positive"
    assert (
        strip_label("20 reviews mention fit: 67% runs small.")
        == "20 reviews mention fit: 67% runs small."
    )
    reply = {"fit_size": "Fit and size: 20 reviews mention fit and 67% say it runs small."}
    res = phrase_summary(_facts(), "Dress", FakeLLM([json.dumps(reply)]))
    assert res["summary"].count("Fit and size:") == 1


def test_phrase_summary_rejects_numbers_borrowed_from_another_aspect() -> None:
    facts = _facts()
    facts["aspects"]["comfort"] = {
        "mentions": 9,
        "pct_of_reviews": 22,
        "pct_positive": 80,
        "pct_negative": 10,
        "pct_mixed": 10,
        "mean_rating": 4.4,
    }
    reply = {
        "fit_size": "20 reviews mention fit; 67% say it runs small.",
        "comfort": "9 reviews mention comfort and 67% say it runs small.",
    }
    res = phrase_summary(facts, "Dress", FakeLLM([json.dumps(reply)]))
    assert res["source"] == "mixed" and res["bad_aspects"] == ["comfort"]


# ------------------------------------------------------------------ Q&A pipeline
def _answer(cites: list[str], text: str = "Runs small [R1][R2].", can: bool = True) -> str:
    return json.dumps(
        {
            "answer": text,
            "can_answer": can,
            "citations": cites,
            "reviewers_agree": "Runs small [R1]",
            "reviewers_disagree": "One says true to size [R3]",
        }
    )


def test_retrieve_scopes_to_item_and_reranks(qa) -> None:
    pipe, _ = qa([])
    ev = pipe.retrieve("A1", "does it run small size")
    assert set(ev["item_id"]) == {"A1"} and ev["review_id"].iloc[0] in {"R1", "R2"}
    assert ev["score"].is_monotonic_decreasing


def test_answer_with_valid_citations(qa) -> None:
    pipe, llm = qa([_answer(["R1", "R2"])])
    res = pipe.answer("A1", "does it run small size")
    out = res.output
    assert not out.refused and set(out.citations) == {"R1", "R2"} and out.evidence_count >= 2
    assert out.aspect == "fit_size" and 0 < out.confidence <= 1 and out.reviewers_disagree
    assert "R1" in llm.calls[0][1]["content"]


def test_invalid_citation_is_dropped_and_recorded(qa) -> None:
    pipe, _ = qa([_answer(["R1", "R99"], text="Runs small [R1] and big [R99].")])
    res = pipe.answer("A1", "does it run small size")
    assert "R1" in res.output.citations and "R99" not in res.output.citations
    assert "R99" in res.invalid_citations and "R99" not in res.output.answer


def test_no_valid_citation_means_refusal(qa) -> None:
    reply = json.dumps(
        {
            "answer": "It runs small.",
            "can_answer": True,
            "citations": ["R98"],
            "reviewers_agree": "",
            "reviewers_disagree": "",
        }
    )
    pipe, _ = qa([reply])
    res = pipe.answer("A1", "does it run small size")
    assert res.output.refused and res.invalid_citations == ["R98"]


def test_weak_evidence_refuses_without_calling_llm(qa) -> None:
    pipe, llm = qa([_answer(["R1"])], min_similarity=0.99)
    res = pipe.answer("A1", "does it run small size")
    assert res.output.refused and res.output.answer == "Not enough reviews mention this."
    assert res.output.citations == [] and llm.calls == []


def test_model_can_decline_and_llm_failure_is_safe(qa) -> None:
    pipe, _ = qa([_answer([], text="Not enough reviews mention this.", can=False)])
    assert pipe.answer("A1", "does it run small size").output.refused
    pipe2, _ = qa([])
    res = pipe2.answer("A1", "does it run small size")
    assert res.output.refused and "no more replies" in (res.error or "")


def test_output_schema_validates() -> None:
    QAAnswer(answer="x", citations=[], confidence=0.5, aspect="general", evidence_count=0)
    with pytest.raises(ValueError):
        QAAnswer(answer="x", citations=[], confidence=1.5, aspect="general", evidence_count=0)


# ------------------------------------------------------------------ evaluation helpers
def test_recall_and_ci_and_validation() -> None:
    assert recall_at_k(["nothing", "runs small"], "runs small", 5) == 1.0
    assert recall_at_k(["nothing", "runs small"], "runs small", 1) == 0.0
    m, lo, hi = mean_ci([1, 1, 0, 1])
    assert lo <= m <= hi
    gt = pd.DataFrame([{"id": "Q1", "kind": "answerable", "item": "A", "n_matching": 1}])
    assert any("only 1" in p for p in validate_questions(gt))


def test_comma_grouped_citations_are_found_and_invalid_ones_stripped(qa) -> None:
    reply = json.dumps(
        {
            "answer": "Runs small [R1, R2, R99].",
            "can_answer": True,
            "citations": [],
            "reviewers_agree": "Small [R1]",
            "reviewers_disagree": "",
        }
    )
    pipe, _ = qa([reply])
    res = pipe.answer("A1", "does it run small size")
    assert set(res.output.citations) >= {"R1", "R2"} and res.invalid_citations == ["R99"]
    assert "R99" not in res.output.answer and "[R1, R2]" in res.output.answer


def test_refusal_phrase_with_can_answer_true_is_a_refusal(qa) -> None:
    pipe, _ = qa([_answer(["R1"], text="Not enough reviews mention this.")])
    assert pipe.answer("A1", "does it run small size").output.refused


def test_strip_invalid_and_extract_ids() -> None:
    from catalogiq.rag.qa import extract_ids, strip_invalid

    assert extract_ids("[R1, R22] and R3", "R1") == ["R1", "R22", "R3"]
    assert strip_invalid("Big [R9] fit [R1, R9].", {"R9"}) == "Big fit [R1]."


def test_split_claims_keeps_cited_ids_and_skips_fragments() -> None:
    from catalogiq.rag.eval_qa import split_claims

    out = {
        "answer": "It runs rather small [R1]. Fine.",
        "reviewers_agree": "Most say the fit is tight [R2][R3].",
        "reviewers_disagree": "None mentioned.",
        "citations": ["R1", "R2", "R3"],
    }
    claims = split_claims(out)
    assert [c["cited"] for c in claims] == [["R1"], ["R2", "R3"]]
    assert claims[0]["text"] == "It runs rather small."
