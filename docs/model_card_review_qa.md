# Model card: review-grounded shopper Q&A (Phase 5)

## What it does
A shopper picks a product and asks a question. The system searches **only that product's reviews**, decides whether the evidence is
strong enough, and if so lets a local LLM answer in validated JSON (`answer`, `citations`, `confidence`, `aspect`, `evidence_count`, plus
"reviewers agree / disagree" lines) with review ids quoted inline. If fewer than 2 distinct reviews are similar enough, it answers
"Not enough reviews mention this." and does not call the LLM. A second feature turns the same reviews into aspect-level product summaries.
Code: `src/catalogiq/rag/{reviews_index,aspects,qa,summary,eval_qa}.py`; prompts: `configs/prompts/`; settings: `configs/qa.yaml`.

## Components
| Part | Choice |
|---|---|
| Index | 139,400 Amazon Fashion reviews -> 166,607 chunks (whole if <= 400 chars, else 3-sentence windows <= 700), ChromaDB collection `amazon_reviews`, `all-MiniLM-L6-v2` |
| Retrieval | within one product: BM25 + vector, reciprocal rank fusion (k=60), re-rank with helpful votes (0.3) and verified purchase (0.2), top 5 |
| Evidence gate | >= 2 distinct reviews with cosine >= 0.52 (stored in `reports/phase5/gate.json`) |
| Answerer | `llama3.2:3b` via Ollama on CPU behind the `LLMClient` interface, JSON-schema output, one repair retry |
| Aspect tagger | keyword rules (six aspects); NLI, embeddings and an LLM were compared and not used |
| Summaries | counts and percentages computed in code; LLM phrases one sentence per aspect, checked per aspect, template fallback |

## Intended use
Help shoppers find what past buyers said about a specific product, with links to the source reviews. **Not** a source of product
specifications, safety or health claims, and not for decisions where a wrong answer is costly without a human check.

## Evaluation (60 questions on 20 products: 45 answerable, 15 unanswerable; one local 3B model)
| Metric | Result |
|---|---|
| Recall@5, hybrid + re-rank (BM25 / vector alone) | 1.00 (0.98 / 0.89); MRR 0.89 |
| Correct refusal of unanswerable questions | 93.3% (14 of 15) |
| Refusal of answerable questions (cost of caution) | 31.1% (14 of 45) |
| Invalid cited review ids | 0 of 99 |
| Cited reviews that contain the answer pattern | 84.8% |
| Fully faithful answers, LLM judge (lower bound) | 40.6% of 32 |
| Hand-check by the project owner (15 answers) | 14 faithful; judge agrees on 9 of 15 (60%); 6 of the judge's 7 flags overturned |
| Latency, answered / refused | 38.5 s / 1.9 s per question (CPU) |
| Product summaries (20 products) | 18 LLM, 1 mixed, 1 template |

Aspect tagger on 100 held-out labelled pieces (macro-F1): NLI 0.585, rules 0.523, embeddings 0.502, LLM 0.427. NLI costs 8.3 s per piece,
so rules are used; the NLI gain has a 95% interval of [-0.007, 0.133].

## Error analysis
- **First pipeline version refused 82% of answerable questions.** The parser ignored comma-grouped ids; fixed, now 31%.
- **Q27 (failure):** "Does the pocket watch come in a gift box?" was answered "Most reviewers say yes" from two reviews. One says "as a gift"
  (matched on the word, not the meaning); the other says "its own special fancy box", which the test label missed. The gate (best match 0.58)
  and the judge both passed it. Next fixes: require the question's key noun in a cited review; state the number of reviews instead of "most".
- **All 14 refusals of answerable questions came from the evidence gate** (fewer than 2 reviews scored >= 0.52 even though a matching review exists); none came from the LLM or from citation checks.
- **Judge too strict:** short summary lines ("The chain or clasp breaks easily") are checked alone against two reviews and fail.

## Limitations
- 60 questions and 20 products; the pipeline was designed while looking at them and the gate threshold was fit on all 60, so figures are optimistic.
- The gate is a weak discriminator (cross-validated about 60% answered / 82% refused); no threshold gives both near 90%.
- The faithfulness judge is the same small model that writes the answers; the human check is 15 answers by one person.
- Aspect labels (200 pieces, one person) are liberal; delivery has 14 positives, so per-aspect scores for it are unreliable.
- Review sentiment in summaries is rating-based, and keyword aspect tags count reviews that mention a topic, not reviews that judge it.
- Reviews are English Amazon Fashion reviews; no measurement on other languages or categories.
- About 39 s per answered question on this CPU is too slow for live use without caching, a faster model or a GPU.

## Reproduce
`scripts/build_review_index.py` -> `scripts/evaluate_qa.py all` (Ollama running) -> `scripts/summarise_qa_eval.py` -> execute `notebooks/05_review_qa.ipynb`.
