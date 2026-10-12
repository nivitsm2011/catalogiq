# Model card: grounded listing generator (Phase 4)

## What it does
Photo -> Phase 2 classifier attributes (with confidence) + optional seller notes -> (optional) retrieval of similar well-rated listings and
the category style guide -> local LLM writes `title`, 5 `bullets`, `description`, `search_tags`, `attributes_used`, `warnings` as validated
JSON -> guardrails check the result. Code: `src/catalogiq/rag/`, `src/catalogiq/llm/client.py`; prompts: `configs/prompts/`.

## Components
| Part | Choice |
|---|---|
| Writer / judge | `llama3.2:3b` via Ollama, CPU, temperature 0.2 (judge 0.0), seed 42, behind the `LLMClient` interface |
| Knowledge base | 800 Amazon Fashion listings + 13 style-guide chunks in ChromaDB (`data/processed/chroma/`) |
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` (384-d, about 90 MB) |
| Retrieval | BM25 + vector, reciprocal rank fusion (k=60), quality boost 0.5, category filter, top 3 examples |
| Guardrails | unsupported materials / claims / numbers / sizes / brands, banned words and health claims, style rules, attribute coverage |

## Intended use
Draft listing copy for a seller to review. **Not** for publishing without human review.

## Evaluation (50 held-out Kaggle test products; stratified by category; seller note = brand only)
| Arm | Listings with unsupported facts | Banned words | Fully style-compliant | Judge clarity / accuracy | Seconds per listing |
|---|---|---|---|---|---|
| no-RAG | 2% | 14% | 0% | 3.32 / 3.70 | 12.8 |
| RAG | 70% | 70% | 2% | 2.86 / 4.26 | 43.8 |
| no-RAG + rewrite | 0% | 2% | 0% | 3.44 / 3.68 | 35.9 |
| RAG + rewrite | 40% | 32% | 16% | 2.84 / 4.14 | 100.6 |

Attribute coverage was 100% in every arm; valid JSON 100%; retrieved examples came from the right category for 96% of products.
Retrieval test (30 queries): precision@5 hybrid 0.78, vector 0.747, BM25 0.707.
Hand-check (10 listings): judge vs human accuracy exact agreement 50% (Spearman 0.00), clarity 20% (-0.33).

## Known failure modes (error analysis)
1. **Fact leakage from examples.** With RAG the 3B model copies materials and claims from example listings (leather in 11 listings,
   "high-quality" in 10, breathable in 6, machine wash in 4). The guardrail flags them; a rewrite step removes about half of the affected listings.
2. **Hype words** ("perfect" in 35 RAG listings) despite the instruction not to use them.
3. **Unsure attributes in titles.** The classifier is often unsure of colour; 31 of 50 RAG titles still contain the unsure colour.
4. **Unfilled placeholders** such as "[material]" in no-RAG bullets, and mostly "please confirm" bullets when little is known.
5. **Wrong inputs give wrong listings.** Human notes found wrong colour/season/usage from the classifier in 6 of 10 hand-checked items.
6. **Judge unreliability.** The judge rated RAG higher on accuracy although the guardrail found far more invented facts in RAG.

## Limitations
50 products and 30 retrieval queries; same model writes and judges; the guardrail word lists are finite (new invented claims can pass, a few
flags are false alarms); the design was iterated on the same 50 products; seller notes did not include materials or sizes; English only;
category groups for Amazon listings come from title keywords (10% unclassified).

## Recommended next steps
Show the model only the style guide and fact-stripped example structure; generate the title from a fixed template; try a larger or hosted
model through `LLMClient`; require seller-entered material and size before generation; re-evaluate on fresh products.
