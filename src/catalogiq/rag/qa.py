"""Shopper Q&A over reviews: per-product hybrid retrieval, re-ranking, evidence gate, cited answer.

Flow: question -> restrict to one product -> BM25 + vector search -> reciprocal rank fusion ->
re-rank by helpfulness and verified purchase -> evidence gate (similarity and number of distinct
supporting reviews) -> local LLM writes a short cited answer -> citations are validated against the
evidence that was shown. Weak evidence never reaches the LLM: the answer is the fixed refusal.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, ValidationError

from catalogiq.config import get_qa_config, resolve_path
from catalogiq.llm.client import LLMClient, LLMError
from catalogiq.logging_utils import get_logger
from catalogiq.rag.aspects import aspect_names, rule_matches
from catalogiq.rag.bm25 import BM25
from catalogiq.rag.knowledge_base import embed_texts
from catalogiq.rag.retrieval import rrf_fuse

logger = get_logger(__name__)

REVIEW_ID = re.compile(r"R\d+")
BRACKET = re.compile(r"\[([^\[\]]*)\]")


def extract_ids(*texts: str) -> list[str]:
    """Review ids (``R123``) found anywhere in the texts, in order, without duplicates."""
    found = [m.group(0) for t in texts for m in REVIEW_ID.finditer(t)]
    return list(dict.fromkeys(found))


def strip_invalid(text: str, invalid: set[str]) -> str:
    """Remove invalid review ids from bracketed citation groups; drop brackets left empty."""

    def fix(m: re.Match[str]) -> str:
        inner = m.group(1)
        ids = REVIEW_ID.findall(inner)
        if not ids or not (set(ids) & invalid):
            return m.group(0)
        keep = [i for i in ids if i not in invalid]
        return f"[{', '.join(keep)}]" if keep else ""

    return re.sub(r"\s{2,}", " ", BRACKET.sub(fix, text)).strip()


class LLMAnswer(BaseModel):
    """What the language model returns (every field is required, so none can be left out)."""

    answer: str = Field(min_length=1)
    can_answer: bool
    citations: list[str]
    reviewers_agree: str
    reviewers_disagree: str


class QAAnswer(BaseModel):
    """Validated final output of the Q&A pipeline."""

    answer: str
    citations: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    aspect: str
    evidence_count: int = Field(ge=0)
    refused: bool = False
    reviewers_agree: str = ""
    reviewers_disagree: str = ""


@dataclass
class QAResult:
    """Pipeline output plus the retrieval trace used by the evaluation."""

    output: QAAnswer
    evidence: pd.DataFrame
    top_similarity: float = 0.0
    latency_s: float = 0.0
    invalid_citations: list[str] = field(default_factory=list)
    raw: str = ""
    error: str | None = None


@lru_cache(maxsize=1)
def _prompts() -> dict[str, str]:
    path = resolve_path(get_qa_config()["paths"]["prompts_qa"])
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def question_aspect(question: str) -> str:
    """Aspect of a question by keyword rules (the aspect with most hits), else ``general``."""
    hits = {a: len(m) for a, m in rule_matches(question).items()}
    best = max(hits, key=hits.get)  # type: ignore[arg-type]
    return best if hits[best] else "general"


class ReviewQA:
    """Answers questions about one product at a time.

    Args:
        llm: Any ``LLMClient``.
        chunks: Chunk table from ``reviews_index.load_chunks``.
        collection: Chroma reviews collection (needed for vector search).
        titles: Optional ``{item_id: product title}`` for the prompt.
        min_similarity: Evidence-gate similarity; defaults to the config value.
    """

    def __init__(
        self,
        llm: LLMClient,
        chunks: pd.DataFrame,
        collection: Any,
        titles: dict[str, str] | None = None,
        min_similarity: float | None = None,
    ) -> None:
        self.cfg = get_qa_config()
        self.rcfg = self.cfg["retrieval"]
        self.llm, self.col, self.titles = llm, collection, titles or {}
        self.min_similarity = (
            self.rcfg["min_similarity"] if min_similarity is None else min_similarity
        )
        self._by_item = {k: v.reset_index(drop=True) for k, v in chunks.groupby("item_id")}
        self._bm25: dict[str, BM25] = {}

    # ------------------------------------------------------------- retrieval
    def _item_bm25(self, item_id: str) -> BM25:
        if item_id not in self._bm25:
            self._bm25[item_id] = BM25(self._by_item[item_id]["text"].tolist(), **self.rcfg["bm25"])
        return self._bm25[item_id]

    def retrieve(
        self,
        item_id: str,
        question: str,
        k: int | None = None,
        mode: str = "hybrid",
        rerank: bool = True,
    ) -> pd.DataFrame:
        """Top chunks of one product.

        Args:
            item_id: Product to search (all other products are excluded).
            question: The shopper's question.
            k: Chunks returned (default ``retrieval.top_k``).
            mode: ``hybrid`` (BM25 + vector, RRF), ``bm25`` or ``vector``.
            rerank: Apply the helpfulness / verified re-rank (otherwise fusion order only).

        Returns:
            Chunk rows with ``similarity`` (question vs chunk cosine), ``rrf`` and ``score``.
        """
        k = k or self.rcfg["top_k"]
        n = self.rcfg["candidate_k"]
        items = self._by_item[item_id]
        q_emb = embed_texts([question])[0]
        bm_scores = self._item_bm25(item_id).scores(question)
        bm_rank = [
            items["chunk_id"].iloc[i] for i in np.argsort(-bm_scores)[:n] if bm_scores[i] > 0
        ]
        vec = self.col.query(
            query_embeddings=[q_emb.tolist()],
            n_results=min(n, len(items)),
            where={"item_id": item_id},
        )
        vec_rank = list(vec["ids"][0])
        rankings = {"hybrid": [bm_rank, vec_rank], "bm25": [bm_rank], "vector": [vec_rank]}[mode]
        fused = rrf_fuse(rankings, self.rcfg["rrf_k"])
        ids = list(fused)
        got = self.col.get(ids=ids, include=["embeddings"])
        emb = {i: np.asarray(e) for i, e in zip(got["ids"], got["embeddings"], strict=True)}
        cand = items.set_index("chunk_id").loc[ids].reset_index()
        cand["rrf"] = [fused[i] for i in ids]
        cand["similarity"] = [float(emb[i] @ q_emb) for i in ids]
        top = cand["rrf"].max() or 1.0
        max_help = math.log1p(max(items["helpful_vote"].max(), 1))
        boost = 1.0
        if rerank:
            boost = (
                1.0
                + self.rcfg["w_helpful"] * np.log1p(cand["helpful_vote"]) / max_help
                + self.rcfg["w_verified"] * cand["verified_purchase"].astype(float)
            )
        cand["score"] = (cand["rrf"] / top) * boost
        return cand.sort_values("score", ascending=False).head(k).reset_index(drop=True)

    def supporting(self, evidence: pd.DataFrame) -> pd.DataFrame:
        """Evidence chunks above the similarity bar, one per review (the best one)."""
        ok = evidence[evidence["similarity"] >= self.min_similarity]
        return ok.sort_values("similarity", ascending=False).drop_duplicates("review_id")

    # ------------------------------------------------------------- confidence
    def confidence(self, support: pd.DataFrame) -> float:
        """Heuristic 0-1 score: mean similarity of the supporting chunks and how many there are."""
        if support.empty:
            return 0.0
        count_part = min(1.0, len(support) / self.rcfg["top_k"])
        return float(
            round(0.5 * min(1.0, support["similarity"].mean() / 0.6) + 0.5 * count_part, 3)
        )

    # ------------------------------------------------------------- answering
    @staticmethod
    def _format_evidence(evidence: pd.DataFrame, max_chars: int) -> str:
        lines = []
        for r in evidence.itertuples():
            ver = "verified" if r.verified_purchase else "unverified"
            lines.append(
                f"[{r.review_id}] ({r.rating:.0f} stars, {ver}, {r.helpful_vote} helpful) "
                f"{r.text[:max_chars]}"
            )
        return "\n".join(lines)

    def _refusal(self, aspect: str, evidence: pd.DataFrame, n_support: int, sim: float) -> QAResult:
        out = QAAnswer(
            answer=self.cfg["qa"]["no_evidence_answer"],
            citations=[],
            confidence=0.0,
            aspect=aspect,
            evidence_count=n_support,
            refused=True,
        )
        return QAResult(out, evidence, sim)

    def answer(self, item_id: str, question: str) -> QAResult:
        """Answer one question about one product, or refuse when the evidence is weak."""
        import time

        t0 = time.perf_counter()
        aspect = question_aspect(question)
        evidence = self.retrieve(item_id, question)
        support = self.supporting(evidence)
        top_sim = float(evidence["similarity"].max()) if len(evidence) else 0.0
        if len(support) < self.rcfg["min_supporting"]:
            res = self._refusal(aspect, evidence, len(support), top_sim)
            res.latency_s = time.perf_counter() - t0
            return res
        shown = support.head(self.rcfg["top_k"])
        qcfg = self.cfg["qa"]
        p = _prompts()
        messages = [
            {"role": "system", "content": p["system"]},
            {
                "role": "user",
                "content": p["user"].format(
                    title=self.titles.get(item_id, item_id),
                    question=question,
                    evidence=self._format_evidence(shown, qcfg["max_chunk_chars_in_prompt"]),
                ),
            },
        ]
        parsed, raw, error = self._ask(messages, qcfg)
        result = QAResult(
            self._refusal(aspect, evidence, len(support), top_sim).output,
            evidence,
            top_sim,
            raw=raw,
            error=error,
        )
        declined = parsed is None or not parsed.can_answer
        if parsed is not None and not declined:
            declined = parsed.answer.strip().lower().rstrip(".") == qcfg[
                "no_evidence_answer"
            ].lower().rstrip(".")
        if parsed is not None and not declined:
            shown_ids = set(shown["review_id"])
            used = extract_ids(
                " ".join(parsed.citations),
                parsed.answer,
                parsed.reviewers_agree,
                parsed.reviewers_disagree,
            )
            valid = [c for c in used if c in shown_ids]
            invalid = {c for c in used if c not in shown_ids}
            result.invalid_citations = sorted(invalid)
            result.output = QAAnswer(
                answer=strip_invalid(parsed.answer, invalid),
                citations=valid,
                confidence=self.confidence(support),
                aspect=aspect,
                evidence_count=len(support),
                refused=False,
                reviewers_agree=strip_invalid(parsed.reviewers_agree, invalid),
                reviewers_disagree=strip_invalid(parsed.reviewers_disagree, invalid),
            )
            if not valid:  # an answer that cites no shown review is not trusted
                result.output = self._refusal(aspect, evidence, len(support), top_sim).output
        result.latency_s = time.perf_counter() - t0
        return result

    def _ask(
        self, messages: list[dict[str, str]], qcfg: dict[str, Any]
    ) -> tuple[LLMAnswer | None, str, str | None]:
        schema = LLMAnswer.model_json_schema()
        convo, raw, error = list(messages), "", None
        for _ in range(1 + qcfg["repair_attempts"]):
            try:
                reply = self.llm.chat(convo, schema=schema, max_tokens=qcfg["max_new_tokens"])
            except LLMError as exc:
                return None, raw, str(exc)
            raw = reply.text
            try:
                return LLMAnswer.model_validate(json.loads(raw)), raw, None
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                error = str(exc)[:300]
                convo = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": _prompts()["repair"].format(error=error)},
                ]
        return None, raw, error


def all_aspects() -> list[str]:
    """Aspect names plus ``general`` (what ``QAAnswer.aspect`` can hold)."""
    return [*aspect_names(), "general"]
