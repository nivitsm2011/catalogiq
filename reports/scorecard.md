# CatalogIQ scorecard

Generated 2026-11-13 19:32 from commit `aee5ade`. The evaluations ran on commit `1e2f664` plus working-tree fixes that were committed afterwards (search and the system test were re-run after two measuring bugs were fixed). Produced by `python -m catalogiq.eval.run_all`; every table below was recomputed in that run unless the Provenance section says otherwise. Experiment `catalogiq-eval` in MLflow, tagged with the commit hash.

## 1. Vision: product attributes from a photo (Phase 2)

Test split: 6,339 photos. Coverage = share of photos whose calibrated confidence clears the attribute's threshold (chosen so that accepted predictions are at least about 90% right on validation).

| Attribute | Macro-F1 (ours) | Majority class | Frozen MobileNet + logreg | Zero-shot CLIP | Accuracy | Confidence threshold | Coverage at threshold | Precision of accepted |
|---|---|---|---|---|---|---|---|---|
| masterCategory | 0.990 | 0.164 | 0.976 | 0.883 | 0.991 | 0.341 | 100.0% | 0.991 |
| subCategory | 0.922 | 0.021 | 0.896 | 0.668 | 0.959 | 0.187 | 100.0% | 0.959 |
| articleType | 0.806 | 0.005 | 0.775 | 0.683 | 0.857 | 0.495 | 92.9% | 0.890 |
| baseColour | 0.582 | 0.025 | 0.469 | 0.561 | 0.675 | 0.765 | 39.8% | 0.885 |
| gender | 0.747 | 0.135 | 0.661 | 0.570 | 0.878 | 0.490 | 96.6% | 0.894 |
| season | 0.714 | 0.164 | 0.679 | 0.232 | 0.699 | 0.791 | 29.1% | 0.903 |
| usage | 0.639 | 0.146 | 0.622 | 0.387 | 0.876 | 0.601 | 92.2% | 0.901 |
| **mean** | 0.771 | 0.094 | 0.725 | 0.569 |  |  |  |  |

Latency (one photo, CPU, preprocessing included): mean 13.9 ms, median 13.1 ms, p95 18.2 ms over 100 runs.

## 2. Visual search and duplicates (Phase 3)

Image queries: 1,995 test photos against a 35,918-photo gallery; a result is relevant if it has the same product type AND colour.

| Representation | Queries | Recall@5 | Recall@10 | Precision@10 | mAP@10 |
|---|---|---|---|---|---|
| CLIP ViT-B/32 image embeddings | 1995 | 0.823 | 0.890 | 0.418 | 0.309 |
| Phase 2 fine-tuned classifier features | 1995 | 0.772 | 0.854 | 0.412 | 0.312 |
| Frozen ImageNet MobileNetV3 features | 1995 | 0.698 | 0.796 | 0.319 | 0.223 |
| Random vectors (floor) | 1995 | 0.040 | 0.088 | 0.010 | 0.003 |

| Text-to-image | Queries | Recall@10 | Precision@10 | mAP@10 |
|---|---|---|---|---|
| CLIP text -> image embeddings | 300 | 0.887 | 0.451 | 0.372 |
| Random gallery items (floor) | 300 | 0.017 | 0.002 | 0.000 |
| CLIP text -> attribute-sentence embeddings (reference, not image-based) | 300 | 0.940 | 0.843 | 0.903 |

Latency over the 42,257-photo catalog: vector search 23.0 ms (34.9 ms filtered); CLIP encodes one photo in 183 ms and one text in 133 ms.

**Duplicate detection** (CLIP similarity and pHash; threshold chosen for at least 95% precision): threshold 0.975, precision 1.00, recall 0.22. Labelled pairs: 38 duplicate / 58 not / 4 unsure. **The labels are provisional (assigned by the AI assistant), and recall rests on very few positives; treat it as unknown.**

## 3. Listing generator: RAG vs no-RAG (Phase 4)

50 held-out products, seller note = brand only. Hallucination = a listing states a material, claim, number or brand that is not in the attributes or seller note (deterministic check). The judge is the same 3B model that writes the listings and agreed poorly with the owner on 10 listings, so it is a secondary metric.

| Arm | Hallucination rate (listings with an unsupported fact) | Attribute coverage | Style compliance | Banned-word rate | Judge clarity / accuracy (1-5) | Seconds per listing |
|---|---|---|---|---|---|---|
| no-RAG | 2.0% | 100.0% | 0.0% | 14.0% | 3.32 / 3.70 | 12.8 |
| RAG | 70.0% | 100.0% | 2.0% | 70.0% | 2.86 / 4.26 | 43.8 |
| no-RAG + rewrite | 0.0% | 100.0% | 0.0% | 2.0% | 3.44 / 3.68 | 35.9 |
| RAG + rewrite | 40.0% | 100.0% | 16.0% | 32.0% | 2.84 / 4.14 | 100.6 |

**RAG vs no-RAG:** hallucination 70.0% vs 2.0%, style compliance 2.0% vs 0.0%, judge accuracy 4.26 vs 3.70, judge clarity 2.86 vs 3.32, time 44 s vs 13 s. Retrieval returned the right category for 96.0% of products. Latencies are the original generation times replayed from the LLM cache.

## 4. Review Q&A (Phase 5)

60 questions on 20 products (45 answerable, 15 unanswerable). Recall@5 = an answer-bearing review is among the top 5.

| Retrieval variant | Recall@5 | MRR | Search ms |
|---|---|---|---|
| BM25 | 0.978 | 0.835 | 215.180 |
| Vector | 0.889 | 0.819 | 180.636 |
| Hybrid (RRF) | 1.000 | 0.829 | 178.103 |
| Hybrid + helpfulness re-rank (used) | 1.000 | 0.893 | 181.848 |

| Metric | Value |
|---|---|
| Correct-refusal rate (15 unanswerable questions) | 93.3% |
| False-refusal rate (45 answerable questions) | 31.1% |
| Citation validity (cited ids that exist among the shown reviews) | 100.0% |
| Citation accuracy (cited review contains the answer pattern) | 84.8% |
| Faithful answers, LLM judge (a LOWER BOUND: the judge is too strict) | 40.6% |
| Faithful answers, owner hand-check of 15 answers (small sample) | 93.3% |
| Judge vs owner agreement on those 15 | 60.0% |
| Latency per answered question / per refusal (CPU, original Phase 5 run) | 38.5 s / 1.9 s |

## 5. Catalog quality checker (Phase 6)

Image checks: 500 clean test photos plus 8,500 degraded copies we made (blur, dark, washed-out, shrunk, stretched, cluttered, text). Precision is strict: a flag raised by a different defect type counts as wrong. Overexposure recall is low because white products on white backgrounds look washed out to a simple measure.

| Check | Precision | Recall | Clean photos flagged |
|---|---|---|---|
| blurry | 0.423 | 0.999 | 2.4% |
| too_dark | 0.899 | 1.000 | 0.2% |
| overexposed | 0.646 | 0.115 | 1.2% |
| low_resolution | 0.667 | 1.000 | 0.0% |
| bad_aspect_ratio | 1.000 | 1.000 | 0.0% |
| cluttered_background | 0.746 | 0.452 | 1.0% |
| text_or_watermark | 0.387 | 0.604 | 2.6% |

Image-text mismatch detection: 300 injected mismatches (100 colour, 100 product type, 100 gender) in the 6,339 test listings. Precision counts untouched listings as false alarms, and the catalog contains real label errors, so it is a lower bound.

| Attribute | Precision | Recall | F1 |
|---|---|---|---|
| baseColour | 0.253 | 0.210 | 0.230 |
| articleType | 0.509 | 0.560 | 0.533 |
| gender | 0.600 | 0.660 | 0.629 |
| overall | 0.472 | 0.477 | 0.474 |

| Signal | Precision | Recall | F1 |
|---|---|---|---|
| classifier alone | 0.529 | 0.420 | 0.468 |
| CLIP alone | 0.198 | 0.433 | 0.272 |
| either | 0.299 | 0.553 | 0.388 |
| both (used) | 0.472 | 0.477 | 0.474 |

Batch report over all 6,339 test listings: 8.3% carry a flag, 4.1% score below 85, mean score 98.7.

## 6. System: end-to-end latency, memory and disk

Photo -> quality check -> attributes -> full listing (title, 5 bullets, description, tags), CPU only, on 5 test photos per pipeline with an **uncached** LLM. Models were loaded once (82 s) and the first LLM call took 7 s (model load).

| Pipeline | Quality check (s) | Attributes (s) | Retrieval + LLM + guardrails (s) | Total mean (s) | Total median (s) | Total p95 (s) |
|---|---|---|---|---|---|---|
| no-RAG | 0.36 | 0.356 | 14.8 | 15.5 | 13.7 | 22.2 |
| RAG + guardrail rewrite | 0.36 | 0.356 | 65.7 | 66.4 | 66.4 | 78.1 |

**Peak RAM:** 5.1 GB in total (1.6 GB the Python process with all models loaded + 3.5 GB the Ollama server holding the language model); the laptop has 16.8 GB.

**Disk footprint:** the project folder is 4.75 GB against a budget of 5 GB. The language model CatalogIQ uses (Ollama stores it outside the folder) adds 2.0 GB, so the whole system is 6.75 GB (over the budget if the model is counted); the Ollama store also holds other models that CatalogIQ does not use (nomic-embed-text:latest 0.3 GB, deepseek-r1:7b 4.7 GB); budget 5 GB.

| Folder | GB |
|---|---|
| .venv | 1.68 |
| data | 2.17 |
| models | 0.85 |
| mlruns | 0.00 |
| reports | 0.02 |
| docs | 0.00 |
| notebooks | 0.01 |
| .git | 0.02 |
| other (src, tests, scripts, configs, ...) | 0.00 |

| Largest items | GB |
|---|---|
| data/processed | 0.92 |
| models/cache | 0.81 |
| data/interim | 0.72 |
| data/raw | 0.53 |

## 7. Drift check against the earlier phases

75 numbers recorded by the earlier phases were compared with this re-run: 68 match, 5 timing, 2 DRIFT. `match` = within the precision the phase recorded; `timing` = a wall-clock number inside a wide band (machine state changes timings); `DRIFT` = outside the tolerance.

| metric | previous | now | diff | status | explanation |
|---|---|---|---|---|---|
| vision.latency_mean_ms | 18.7 | 13.92 | -4.78 | timing | Faster than recorded (13.9 ms vs 18.7 ms mean). Wall-clock latency depends on machine state; same function and 100 runs. Accuracy numbers are identical. |
| vision.latency_p95_ms | 22.2 | 18.16 | -4.039 | timing | Faster than recorded (13.9 ms vs 18.7 ms mean). Wall-clock latency depends on machine state; same function and 100 runs. Accuracy numbers are identical. |
| search.vector_search_ms | 8.2 | 23.02 | 14.82 | DRIFT | About 3x slower than in Phase 3, identical in two runs an hour apart. Not the harness: timing CLIP standalone gave the same slowdown. Windows reported the laptop on battery power, which probably throttles the CPU. Retrieval metrics are unaffected; re-measure plugged in. |
| search.clip_encode_image_ms | 60 | 182.8 | 122.8 | DRIFT | About 3x slower than in Phase 3 (183 ms vs 60 ms); a standalone check gave 188 ms, so it is the machine state (laptop on battery), not the code. Re-measure plugged in. |
| listing.no_rag.latency_mean_s | 12.8 | 12.84 | 0.04183 | timing | Original generation time replayed from the LLM cache (the number is stored with the reply), so it matches by construction; the tiny difference is rounding in the 2-decimal record. |
| listing.rag.latency_mean_s | 43.8 | 43.84 | 0.03555 | timing | Original generation time replayed from the LLM cache (the number is stored with the reply), so it matches by construction; the tiny difference is rounding in the 2-decimal record. |
| qa.latency_answered_s | 38.5 | 38.52 | 0.01789 | timing | Taken from the original Phase 5 run (cached replies replay instantly); the system section re-measures real speed with an uncached model. |

<details><summary>All compared numbers</summary>

| metric | previous | now | diff | status | source |
|---|---|---|---|---|---|
| vision.masterCategory.macro_f1 | 0.99 | 0.9899 | -8.967e-05 | match | progress.md Phase 2 |
| vision.subCategory.macro_f1 | 0.922 | 0.9216 | -0.0004055 | match | progress.md Phase 2 |
| vision.articleType.macro_f1 | 0.806 | 0.806 | -4.385e-05 | match | progress.md Phase 2 |
| vision.baseColour.macro_f1 | 0.582 | 0.5821 | 8.523e-05 | match | progress.md Phase 2 |
| vision.gender.macro_f1 | 0.747 | 0.7474 | 0.0003589 | match | progress.md Phase 2 |
| vision.season.macro_f1 | 0.714 | 0.7141 | 9.207e-05 | match | progress.md Phase 2 |
| vision.usage.macro_f1 | 0.639 | 0.6391 | 0.0001224 | match | progress.md Phase 2 |
| vision.mean.macro_f1 | 0.771 | 0.7714 | 0.0004456 | match | progress.md Phase 2 |
| vision.articleType.accuracy | 0.857 | 0.8566 | -0.000398 | match | progress.md Phase 2 |
| vision.baseColour.accuracy | 0.675 | 0.675 | 2.761e-05 | match | progress.md Phase 2 |
| vision.logreg.mean.macro_f1 | 0.725 | 0.7254 | 0.0003608 | match | progress.md Phase 2 |
| vision.clip_zero_shot.mean.macro_f1 | 0.569 | 0.5692 | 0.0002194 | match | progress.md Phase 2 |
| vision.baseColour.coverage | 0.4 | 0.3983 | -0.001672 | match | progress.md Phase 2 open issues (40%) |
| vision.season.coverage | 0.29 | 0.2912 | 0.001239 | match | progress.md Phase 2 open issues (29%) |
| vision.latency_mean_ms | 18.7 | 13.92 | -4.78 | timing | progress.md Phase 2 |
| vision.latency_p95_ms | 22.2 | 18.16 | -4.039 | timing | progress.md Phase 2 |
| search.clip.recall@5 | 0.823 | 0.8231 | 0.0001 | match | progress.md Phase 3 |
| search.clip.recall@10 | 0.89 | 0.8897 | -0.0003 | match | progress.md Phase 3 |
| search.clip.precision@10 | 0.418 | 0.4182 | 0.0002 | match | progress.md Phase 3 |
| search.clip.map@10 | 0.309 | 0.3092 | 0.0002 | match | progress.md Phase 3 |
| search.clf_features.recall@10 | 0.854 | 0.8536 | -0.0004 | match | progress.md Phase 3 |
| search.frozen.recall@10 | 0.796 | 0.796 | 0 | match | progress.md Phase 3 |
| search.text.recall@10 | 0.887 | 0.8867 | -0.0003 | match | progress.md Phase 3 |
| search.text.map@10 | 0.372 | 0.3718 | -0.0002 | match | progress.md Phase 3 |
| search.vector_search_ms | 8.2 | 23.02 | 14.82 | DRIFT | progress.md Phase 3 |
| search.clip_encode_image_ms | 60 | 182.8 | 122.8 | DRIFT | progress.md Phase 3 |
| duplicates.threshold | 0.975 | 0.975 | 0 | match | progress.md Phase 3 (AI-provisional labels) |
| duplicates.precision | 1 | 1 | 0 | match | progress.md Phase 3 (AI-provisional labels) |
| duplicates.recall | 0.22 | 0.2192 | -0.0008 | match | progress.md Phase 3 (AI-provisional labels) |
| listing.no_rag.hallucination_rate | 0.02 | 0.02 | 0 | match | progress.md Phase 4 |
| listing.rag.hallucination_rate | 0.7 | 0.7 | 0 | match | progress.md Phase 4 |
| listing.no_rag_fix.hallucination_rate | 0 | 0 | 0 | match | progress.md Phase 4 |
| listing.rag_fix.hallucination_rate | 0.4 | 0.4 | 0 | match | progress.md Phase 4 |
| listing.no_rag.banned_rate | 0.14 | 0.14 | 0 | match | progress.md Phase 4 |
| listing.rag.banned_rate | 0.7 | 0.7 | 0 | match | progress.md Phase 4 |
| listing.no_rag.style_compliance | 0 | 0 | 0 | match | progress.md Phase 4 |
| listing.rag.style_compliance | 0.02 | 0.02 | 0 | match | progress.md Phase 4 |
| listing.rag_fix.style_compliance | 0.16 | 0.16 | 0 | match | progress.md Phase 4 |
| listing.no_rag.judge_clarity | 3.32 | 3.32 | 0 | match | progress.md Phase 4 |
| listing.rag.judge_clarity | 2.86 | 2.86 | 0 | match | progress.md Phase 4 |
| listing.no_rag.judge_accuracy | 3.7 | 3.7 | 0 | match | progress.md Phase 4 |
| listing.rag.judge_accuracy | 4.26 | 4.26 | 0 | match | progress.md Phase 4 |
| listing.rag.retrieval_hit_rate | 0.96 | 0.96 | 0 | match | progress.md Phase 4 |
| listing.no_rag.latency_mean_s | 12.8 | 12.84 | 0.04183 | timing | progress.md Phase 4 (original generation time kept in the LLM cache) |
| listing.rag.latency_mean_s | 43.8 | 43.84 | 0.03555 | timing | progress.md Phase 4 (original generation time kept in the LLM cache) |
| qa.retrieval.bm25.recall5 | 0.978 | 0.9778 | -0.0002222 | match | progress.md Phase 5 |
| qa.retrieval.vector.recall5 | 0.889 | 0.8889 | -0.0001111 | match | progress.md Phase 5 |
| qa.retrieval.hybrid_rerank.recall5 | 1 | 1 | 0 | match | progress.md Phase 5 |
| qa.retrieval.hybrid_rerank.mrr | 0.893 | 0.893 | -3.704e-05 | match | progress.md Phase 5 |
| qa.correct_refusal_rate | 0.933 | 0.9333 | 0.0003333 | match | progress.md Phase 5 |
| qa.false_refusal_rate | 0.311 | 0.3111 | 0.0001111 | match | progress.md Phase 5 |
| qa.citation_relevance | 0.848 | 0.8485 | 0.0004848 | match | progress.md Phase 5 |
| qa.citation_validity | 1 | 1 | 0 | match | progress.md Phase 5 (0 of 99 invalid) |
| qa.faithful_answers | 0.406 | 0.4062 | 0.00025 | match | progress.md Phase 5 (LLM judge, lower bound) |
| qa.claim_support_rate | 0.567 | 0.5672 | 0.0001642 | match | progress.md Phase 5 |
| qa.handcheck_agreement | 0.6 | 0.6 | 0 | match | progress.md Phase 5 (owner hand-check, n=15) |
| qa.latency_answered_s | 38.5 | 38.52 | 0.01789 | timing | progress.md Phase 5 (original generation time kept in the LLM cache) |
| quality.blurry.recall | 0.999 | 0.999 | 0 | match | progress.md Phase 6 |
| quality.blurry.precision | 0.423 | 0.4235 | 0.0004845 | match | progress.md Phase 6 |
| quality.too_dark.recall | 1 | 1 | 0 | match | progress.md Phase 6 |
| quality.too_dark.precision | 0.899 | 0.8993 | 0.0002806 | match | progress.md Phase 6 |
| quality.overexposed.recall | 0.115 | 0.115 | 0 | match | progress.md Phase 6 |
| quality.cluttered_background.recall | 0.452 | 0.452 | 0 | match | progress.md Phase 6 |
| quality.cluttered_background.precision | 0.746 | 0.7459 | -0.0001254 | match | progress.md Phase 6 |
| quality.text_or_watermark.recall | 0.604 | 0.604 | 0 | match | progress.md Phase 6 |
| quality.text_or_watermark.precision | 0.387 | 0.3872 | 0.0001795 | match | progress.md Phase 6 |
| quality.low_resolution.recall | 1 | 1 | 0 | match | progress.md Phase 6 |
| quality.bad_aspect_ratio.recall | 1 | 1 | 0 | match | progress.md Phase 6 |
| consistency.overall.precision | 0.472 | 0.4719 | -5.281e-05 | match | progress.md Phase 6 |
| consistency.overall.recall | 0.477 | 0.4767 | -0.0003333 | match | progress.md Phase 6 |
| consistency.baseColour.f1 | 0.23 | 0.2295 | -0.0004918 | match | progress.md Phase 6 |
| consistency.articleType.f1 | 0.53 | 0.5333 | 0.003333 | match | progress.md Phase 6 |
| consistency.gender.f1 | 0.63 | 0.6286 | -0.001429 | match | progress.md Phase 6 |
| quality.batch.pct_with_any_issue | 8.3 | 8.314 | 0.01361 | match | progress.md Phase 6 |
| quality.batch.pct_needs_attention | 4.1 | 4.117 | 0.01737 | match | progress.md Phase 6 |

</details>

## 8. Product impact (assumptions are labelled)

**The owner's timings of manual listings have NOT been supplied yet.** Every time and volume below is a placeholder ASSUMPTION, so every derived business number is ILLUSTRATIVE and must not be quoted as a result. Time 5 manual listings with `reports/phase7/listing_timing.html`, save the CSV as `reports/phase7/manual_listing_times.csv`, and run `python -m catalogiq.eval.run_all --render-only`.

**Inputs**

| input | value | unit | status | basis |
|---|---|---|---|---|
| Time to write one listing by hand | 12.0 | min | ASSUMPTION | configs/eval.yaml (placeholder) |
| Time to review and fix one generated draft | 3.0 | min | ASSUMPTION | configs/eval.yaml (placeholder) |
| Ops spot check of one listing | 0.5 | min | ASSUMPTION | configs/eval.yaml (placeholder) |
| Ops review of one flagged listing | 2.0 | min | ASSUMPTION | configs/eval.yaml (placeholder) |
| Listings per month (illustrative marketplace) | 10,000.0 | listings | ASSUMPTION | configs/eval.yaml (placeholder) |
| Labour cost per hour (illustrative) | 10.0 | currency units | ASSUMPTION | configs/eval.yaml (placeholder) |

**Conversions**

| quantity | value | status | how |
|---|---|---|---|
| Machine time from photo to draft listing (CPU) | 16 s (no-RAG) to 66 s (RAG with rewrite) | MEASURED | end-to-end run, uncached LLM (system section) |
| Hands-on minutes saved per listing | 9.0 min (12.0 manual - 3.0 review) | ILLUSTRATIVE (depends on assumptions) | manual time minus review time; machine time is waiting, not hands-on, time |
| Hands-on hours saved per month | 1,500 h at 10,000 listings | ILLUSTRATIVE (depends on assumptions) | minutes saved x listings per month (volume is an assumption) |
| Labour cost saved per month | 15,000 currency units at 10 per hour | ILLUSTRATIVE (depends on assumptions) | hours saved x hourly cost (both assumptions) |
| Share of listings the quality checker flags / sends to review | 8.3% any flag, 4.1% score below 85 | MEASURED | batch report, 6,339 test listings (includes false alarms; misses unknown) |
| Ops review hours per month: look at flagged only vs spot check all | 28 h vs 83 h | ILLUSTRATIVE (depends on assumptions) | volume x rate x minutes per review (volume and minutes are assumptions) |
| RAG drafts with an unsupported fact | 70% | MEASURED | Phase 4 guardrails, 50 products; a person must still review drafts |
| no-RAG + rewrite drafts with an unsupported fact | 0% | MEASURED | Phase 4 guardrails, 50 products; a person must still review drafts |
| Shopper questions: answered with citations / refused when no evidence | 69% of answerable answered, 93% of unanswerable refused | MEASURED | Phase 5, 60 questions |

**Not estimated (no data to support it):**
- Change in returns: no returns data exists in this project.
- Change in conversion or search revenue: needs an A/B test on a real marketplace.
- Shopper time saved by Q&A: needs real question traffic, not the 60 test questions.
- Seller satisfaction: no seller was asked.

## 9. Limitations and risks

### Dataset bias
- **One marketplace's catalog.** The photos and attributes come from the Kaggle "Fashion Product Images (Small)" dataset, an Indian fashion marketplace's catalog. It mixes Indian ethnic wear (kurtas, sarees) with Western casual wear and is dominated by men's and women's items (Men 3,216 and Women 2,619 of the 6,339 test photos; Unisex 283, Boys 127, Girls 94). Results say nothing about other regions, price tiers or categories.
- **Studio photos at 60x80 pixels.** Every photo is a clean studio thumbnail. The gap to what small sellers upload with a phone (the problem CatalogIQ is meant to solve) is **unmeasured**: only 2 of 120 natural photos had a visible defect, and defect detection was measured on copies we degraded ourselves.
- **Two unrelated sources are combined.** Listing examples and reviews come from a sample of Amazon Fashion (800 items, 139,400 reviews, English, mostly jewellery and accessories), while photos and attributes come from the Kaggle catalog. The review Q&A was never tested on the Kaggle products.
- **Reviews are self-selected** and skew to extremes and verified buyers; "what reviewers say" is not "what buyers think".

### Label noise
- **Colour is noisy:** twin photos of the same product disagree on colour 44% of the time, and the classifier is 67.5% accurate on colour. A manual review found about 10% of articleType "errors" were wrong catalog labels, and 12 of 24 untouched listings flagged by the consistency check looked like real catalog errors.
- **Several labels are the AI assistant's, not a person's:** the 100 duplicate pairs (Phase 3), the 120 natural photos and the 24-listing review (Phase 6). The owner's 200 aspect labels are liberal (quality ticked on 144 of 200), which depresses every aspect score.
- **Test answers come from word patterns** written by the developer (Phase 5), which are repeatable but imperfect (one question was mislabelled unanswerable).

### Small-LLM limits
- **A 3B model copies facts from examples.** With retrieved example listings, 70% of generated listings contained an unsupported material, claim or number; the deterministic guardrails, not the model, are what make the output safe, and even then 40% remained after a rewrite.
- **The judge is the same model as the writer.** On the owner's hand-checks it agreed poorly (listings: Spearman 0.00 on accuracy, n=10; answers: 60% agreement, n=15, and it was too strict). Judge scores are secondary metrics and faithfulness percentages from the judge are lower bounds.
- **It over-generalises** ("most reviewers say" from two reviews) and the evidence gate is a weak discriminator: 31% of answerable questions were refused.
- **It is slow on CPU:** seconds to minutes per listing and about 40 s per answered question, too slow for interactive use without caching, a smaller model or a GPU.
- **Review text is untrusted input to the model.** Prompt injection through review text was not tested.

### How far to trust the numbers
- The test sets are small (50 products, 60 questions, 100 duplicate pairs, 120 labelled photos) and most numbers have no confidence interval or repeated seeds.
- Phases 4 and 5 were designed and iterated while looking at the same test sets, and the Q&A gate threshold was fit on all 60 questions, so those results are optimistic.
- Several metrics are proxies: "relevant" in visual search means same product type AND colour; consistency precision counts real catalog errors as false alarms; defects are scored on degraded copies and injected mismatches.
- All hand-checks are by one person. The evaluation was designed and run by the same AI assistant that built the system.

### What would change at production scale
- **Search:** exact search is fine up to about 400,000 items on this CPU; beyond that use HNSW. Embedding 42,000 photos took about 2 hours of CPU, so a real catalog needs incremental embedding and probably a GPU.
- **Reviews:** 166,607 chunks become millions; the vector store, the in-house BM25 and the per-product filter would need a production search service.
- **LLM:** a 3B CPU model would be replaced by a faster or hosted model behind the same `LLMClient` interface, and every Phase 4 and 5 result (hallucination, refusal, faithfulness) would have to be re-measured with an independent judge. Cost, latency and data privacy would become decisions.
- **Thresholds and drift:** quality thresholds must be set per category (jewellery and watches are over-flagged by global thresholds), and calibration, flag rates and refusal rates need monitoring as the catalog changes.
- **Humans in the loop:** flagged listings need a review queue and a feedback loop that produces better labels; auto-publishing generated copy without review is not supported by these results.
- **Business impact must be measured, not modelled:** returns, conversion and search revenue need an A/B test on a real marketplace. Nothing in this project estimates them.


## 10. Provenance

Fixed test sets:
- **vision**: Phase 1 test split, 6,339 photos (group-aware split, seed 42)
- **search**: 1,995 test photos as queries against the 35,918-photo train+val gallery (seed 42); 300 attribute text queries
- **listing**: 50 held-out test products stratified by category group (seed 42), seller note = brand only
- **qa**: 60 questions on 20 products (45 answerable, 15 unanswerable), configs/qa_test_questions.yaml
- **quality**: 500 clean test photos + 8,500 degraded copies (seed 43); 300 injected mismatches in the 6,339 test listings (seed 49)

| Component | Seconds | Recomputed | Cached inputs / replayed | LLM calls that missed the cache | Caveat |
|---|---|---|---|---|---|
| vision | 31 | classifier inference on all test photos, scoring, coverage, latency | baseline predictions (majority recomputed; logistic regression and CLIP zero-shot predictions cached from Phase 2 and re-scored) |  |  |
| search | 156 | retrieval over cached CLIP embeddings, metrics, latency | CLIP embeddings of the 42,257 photos (about 2 h of CPU to recompute) |  |  |
| duplicates | 1 | threshold rule on the 100 labelled pairs |  |  | PROVISIONAL: assigned by the AI assistant, not by a person |
| listing | 357 | product sample, classifier attributes, retrieval, guardrails, judge scoring | replies read from the LLM cache (a cached reply keeps its original latency) | 0 |  |
| qa | 64 | retrieval variants, gate tuning, answers, judge, citation checks | replies read from the LLM cache | 0 | faithfulness hand-check of 15 answers by the owner (saved ratings) |
| quality | 807 | degraded copies and flags, consistency injection and scoring (thresholds kept as tuned), batch report over all test listings |  |  | natural-photo labels and the 24-listing review are the AI assistant's |
