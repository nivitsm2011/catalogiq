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
