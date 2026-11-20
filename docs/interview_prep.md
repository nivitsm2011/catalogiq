# Interview prep: CatalogIQ

Results come from [reports/scorecard.md](../reports/scorecard.md). A few design figures (model size, ECE, split proportions, the 5.4M parameter count, the 18.7 ms first latency measurement) come from [decisions.md](decisions.md) and the [classifier model card](model_card_classifier.md), the repo's own records; none are invented. Where something is **not** verified or measured, the answer says so: that honesty is part of the story. Say these in your own words; they are notes, not a script.

## 1. The 60-second pitch

> "CatalogIQ is an AI copilot for online marketplaces, built end to end on a laptop CPU. Small sellers upload poor photos and thin listings, which hurts search and returns, and shoppers can't get quick answers. I built three sides of it. For sellers, a classifier I trained predicts seven product attributes from a photo, a local 3-billion-parameter LLM writes the listing, and a fact-check catches claims the seller never made. For shoppers, CLIP search and a review Q&A that cites real reviews or says it lacks evidence. For catalog teams, a 0-to-100 quality score with plain-English fixes.
>
> What I'm proudest of is the evaluation. One command re-runs every test on a fixed test set and writes a scorecard, including the failures: retrieved examples made the small model invent facts in 70% of listings, so I built deterministic guardrails that brought it to zero in my test. And I wrote down what I can't claim: small studio photos, small test sets, nothing about returns or revenue."

## 2. The 5-minute deep dive

**0:00 Problem and users (30 s).** Three users: seller, shopper, catalog team. The brief's constraint set the shape: everything local, CPU only, under about 5 GB, a local LLM behind an interface.

**0:30 Data and the split (45 s).** Kaggle Fashion Product Images (about 44k photos, 60x80 pixels) and a sample of Amazon Fashion reviews (800 items, 139,400 reviews). Cleaning decisions were made for business reasons: colours collapsed to 15 retail families because shoppers filter by family, rare classes dropped because a seller tool should not promise a type it has seen under 100 times. The key decision: split by near-duplicate photo group, so a re-shot copy of a training photo can never sit in the test set.

**1:15 Vision (45 s).** MobileNetV3-Large with seven heads, class-weighted loss, no hue augmentation because it would corrupt the colour label. Calibrated confidences, and a threshold per attribute so that accepted predictions are about 90% right on validation. Result: mean macro-F1 0.771 against 0.094 for the majority class, 13.9 ms per photo. Colour and season are weak, so the app asks the seller to confirm them instead of auto-filling.

**2:00 Search and quality (45 s).** CLIP zero-shot beat my fine-tuned features on Recall@10 (0.890 against 0.854) without training. The quality checker is simple OpenCV measures with thresholds tuned by a rule fixed beforehand (maximum recall while flagging at most 3% of clean photos), plus CLIP for text and watermarks.

**2:45 RAG and the surprise (75 s).** Listing generator: hybrid retrieval, style rules, a 3B model, then code that checks for unsupported materials, claims, numbers and brands. A 4-arm ablation on 50 products showed retrieval hurt: 70.0% of listings had an unsupported fact with examples, 2.0% without, 0.0% without plus a rewrite. The small model copies from the examples. So the app defaults to the safe pipeline. Review Q&A: product-scoped hybrid retrieval, an evidence gate that refuses before spending 40 seconds of CPU, citations validated in code: 93.3% correct refusals, 100% valid citations, and an honest cost of 31.1% false refusals.

**4:00 Engineering (45 s).** FastAPI with validation, rate limits and friendly errors; a Streamlit app with a demo mode; MLflow; 222 tests at 75.5% coverage measured the way CI sees it; a drift check that re-ran 75 earlier numbers (68 matched). Docker files are written but I have not built them, and I say so.

**4:45 What I'd do next (15 s).** Real phone photos, an independent judge, a better evidence gate, and an A/B test, because nothing here measures returns or conversion.

## 3. Thirty likely questions

### Computer vision

**1. Why transfer learning and why MobileNetV3?**
The data is 29,579 training photos of 60x80 pixels and the budget is a CPU laptop, so training from scratch was not sensible. MobileNetV3-Large is about 5.4M parameters, exports to a 13 MB TorchScript file and ran at 18.7 ms per photo when I measured it (13.9 ms in the final scorecard). I did not benchmark EfficientNet-B0, so I say the choice rests on published size and measured MobileNet speed, not on a head-to-head test. I trained in two stages: heads only for 4 epochs, then fine-tuning for 6.

**2. How did you handle class imbalance?**
Three layers. First, data: classes under 100 photos were dropped or merged into "Other" depending on the attribute (a form field can legitimately say "Other"; the core taxonomy cannot). Second, the loss: weights proportional to (1/count)^0.5, a middle ground, rather than oversampling, because the data has near-duplicate photos and oversampling encourages memorising them. Third, the metric: I report macro-F1 against a majority-class baseline (0.094) so a model cannot look good by predicting the big class.

**3. What is calibration, how did you do it, and why does it matter here?**
A model is calibrated when "80% confident" is right about 80% of the time. The product needs it because the app auto-fills a value only when confidence clears a threshold; otherwise it asks the seller. I fitted one temperature per attribute on the validation split and set each threshold to the lowest confidence at which accepted validation predictions are at least 90% right. The networks were already well calibrated (ECE between 0.0018 and 0.0239 before scaling), so scaling helped modestly. On test, precision of accepted predictions was 88.5% to 90.3% for the attributes where the threshold actually rejects photos (product type, colour, gender, season, usage), so the 90% target is approximately met, not exactly. The cost is coverage: colour is accepted for only 39.8% of photos and season for 29.1%.

**4. What did Grad-CAM tell you?**
I used it as a qualitative check, on the last feature block (an 8x6 map for a 128x96 input). For correct predictions it highlights the product, but in the T-shirt example part of the attention fell on the model's face and the background. The map is coarse because the photos are tiny, so I treat it as a sanity aid, not proof. It did not change a decision.

**5. How did you prevent leakage, and did it work?**
Many catalog photos are near-duplicates (the same product re-shot). I grouped photos by perceptual hash (Hamming distance at most 1) and split by group, stratified by product type, giving exactly 70/15/15 with zero groups crossing splits (checked in the pipeline and by a unit test). My first threshold, 2, chained unrelated shoes into one 2,859-image cluster, which made a 70/15/15 split impossible. Later I found that CLIP-based duplicates still cross splits for 71 test photos (1.1%); removing them moved articleType accuracy from 85.7% to 85.5%, so the earlier results stand, and I noted that future splits should group by CLIP duplicates too.

**6. Colour macro-F1 is only 0.582. Why, and what would you do?**
Mostly label noise and ambiguity: twin photos of the same product disagree on colour 44% of the time (partly legitimate colourways), and I collapsed 47 raw colours to 15 families to make it learnable. Accuracy is 67.5%. The product answer was design, not a better model: colour is a "please confirm" suggestion. To improve it I would need cleaner colour labels, larger photos, and probably a colour-specific head or histogram features.

### Embeddings and vector search

**7. Why CLIP zero-shot instead of your own fine-tuned features?**
I measured it: CLIP image embeddings reached Recall@5 0.823 and Recall@10 0.890 against 0.772 and 0.854 for my classifier's features and 0.698 and 0.796 for frozen ImageNet features, with no training. Random vectors give 0.088 at 10, which shows the metric is not trivially easy. CLIP is also what makes text-to-image search work (Recall@10 0.887).

**8. Exact or approximate index?**
Exact (`IndexFlatIP`) by default. At 42k items it is a few milliseconds with perfect recall; HNSW was 0.18 ms at 99.9% recall. Extrapolating on synthetic catalogs, exact search crosses a 50 ms budget at about 400,000 items on this CPU, which is where I would switch. Filtered queries restrict the candidates first and search exactly, so a filter never returns fewer results than exist, which post-filtering an approximate index can.

**9. How did you evaluate search, and what are the weak points?**
Queries were 1,995 test photos against a 35,918-photo train+val gallery, and an item is relevant if it has the same product type and the same colour. Recall@k is a hit rate (at least one relevant item in the top k), because hundreds of items are relevant per query. The weak points: "relevant" is a proxy, colour labels are noisy, and the photos are all clean studio shots. My last latency numbers (vector search 23.0 ms, CLIP encode 183 ms) are about 3x slower than first measured; I flagged them as drift and attribute them to battery power, which I did not verify.

**10. How does duplicate detection work, and how much do you trust it?**
A pair is flagged when CLIP cosine is at least 0.975 and the perceptual-hash distance is at most 8. I chose the lowest threshold with at least 95% weighted precision on 100 labelled pairs, because wrongly hiding a legitimate listing is worse than missing a duplicate. Precision came out 1.00 and recall 0.22. I would not quote the recall: the labels were assigned by the AI assistant, not a person, and there are only 38 duplicates. The app keeps a human in the loop (merge or keep both).

### RAG

**11. How did you chunk the reviews?**
Most reviews are short (mean 172 characters), so reviews up to 400 characters stay whole, which keeps context. Longer ones become three-sentence windows with a stride of two, so a single remark is not diluted, with the review title on the first chunk. Only 9,532 reviews (7%) needed splitting; 139,400 reviews became 166,607 chunks, with the metadata (rating, votes, verified purchase) kept for re-ranking.

**12. Why hybrid retrieval, and how do you combine the rankings?**
Questions mix exact words ("zipper") and paraphrases ("does it fit large"). On my 45 answerable questions, BM25 reached Recall@5 0.978, vectors 0.889 and hybrid 1.000. I merge with reciprocal-rank fusion (k=60) because it needs no score calibration, then re-rank with helpful votes and verified purchase, which raised MRR from 0.83 to 0.89. Retrieval is restricted to the product on screen. Caveat: Recall@5 is judged by regex patterns I wrote, so it is lenient.

**13. How do you control hallucination?**
Four layers. The prompt states the facts and marks uncertain attributes. The output is structured JSON validated by Pydantic. A deterministic checker compares the listing with the attributes and seller notes only (never the retrieved examples) for materials, claims, numbers and brands, using word lists in config, plus banned words and style limits. If it finds problems, the model rewrites once with the problems listed, and the system keeps the better draft. Result on 50 products: 70.0% unsupported-fact rate with RAG, 2.0% without, 0.0% without plus rewrite. The limit: word lists are finite, so a novel claim can slip through.

**14. How did you evaluate the generator, and what can't you trust?**
Four arms (RAG or not, rewrite or not) on 50 held-out products with the same checks. The deterministic checks are the main metrics. I also used an LLM judge, but it is the same 3B model that writes the listings and agreed poorly with my hand-check (Spearman 0.00 on accuracy, 10 listings), so it is secondary. Another limit: I iterated the design (fixing two metric bugs, adding the rewrite arm) while looking at the same 50 products, and the notebook says so.

**15. Your Q&A refuses 31% of answerable questions. Is that acceptable?**
It is a deliberate trade-off I would expose as a setting. The gate requires at least two distinct reviews above a similarity of 0.52; otherwise a fixed refusal is returned without calling the model. That refused 93.3% of unanswerable questions but also 31.1% of answerable ones. The planned approach (tune on 24 dev questions, test on 36) did not generalise, so I chose the threshold by product-grouped cross-validation and fitted the final value on all 60 questions, which makes the end-to-end numbers slightly optimistic. For a shopper, a wrong confident answer is worse than "not enough reviews mention this", so I accepted the cost and documented it.

**16. How do you know the citations are right?**
Two checks. Validity: every cited review id must be among the reviews shown to the model (100%, none of 99 invalid). Accuracy: whether the cited review contains the answer pattern (84.8%). An answer with no valid citation becomes a refusal. For faithfulness I used a claim-level LLM judge (40.6%, a lower bound because it is too strict) and checked 15 answers by hand: 14 were faithful (93.3%), a small sample.

### LLM choice and local deployment

**17. Why a local 3B model?**
The brief required everything local and CPU-only, and a local model means no data leaves the machine and no per-call cost. The price is quality and speed: about 15.5 s (p95 22 s) per listing without RAG and 66 s with RAG and rewrite, about 40 s per answered question, and the model copies from examples. It needs about 3.5 GB of RAM in the Ollama server on top of 1.6 GB for my models.

**18. How is the model swappable?**
Everything calls `LLMClient.chat`. `OllamaClient` implements it, `CachedClient` wraps any client with a disk cache keyed on the full request, and a provider table in the config chooses the backend. A hosted model is one more class. The cache has a second job: it makes the 4-arm evaluation repeatable and the demo products instant. In the evaluation harness, replaying through the cache had zero misses and identical metrics.

**19. Your judge is the same model as the writer. What did you do about it?**
I treated it as a secondary signal, reported when it disagreed with me, and added a human hand-check (10 listings, 15 answers). For faithfulness I report the judge's 40.6% explicitly as a lower bound next to the hand-check's 93.3%. The fix I did not do: use an independent, stronger judge. It is in the future-work list.

### Product thinking

**20. Who is the user and what would success look like?**
Sellers need less time to a correct listing; shoppers need trustworthy answers; catalog teams need to review fewer listings. In production I would track listing completion time and edit rate, the share of auto-filled values sellers change, search click-through and zero-result rate, and return rate by flagged versus unflagged listings. None of those is measured here. The scorecard's impact section uses placeholder assumptions and labels the numbers ILLUSTRATIVE; I planned to replace them with my own timing of five manual listings.

**21. What would you A/B test?**
(1) Assisted versus manual listing on a real marketplace: time to publish, edit rate, search impressions. (2) Auto-fill versus "please confirm" at different confidence thresholds. (3) Quality-score nudges to sellers, measured by fix rate and later returns. (4) Review Q&A with refusal versus a more permissive gate, measured by question-to-purchase and complaint rate. Each needs real traffic, which is exactly what this project lacks.

**22. How did you decide when to auto-fill and when to ask?**
The 90%-precision rule per attribute, plus a product rule: auto-fill only when coverage at that bar is at least 90% and macro-F1 is at least 0.70 (masterCategory, subCategory, articleType, gender). Colour, season and usage are always suggestions. A coverage-only rule would have auto-filled usage, which is only about 10 accuracy points above always guessing "Casual".

**23. RAG did worse than no-RAG. How do you ship that?**
Honestly: the app defaults to the safe pipeline (no examples plus rewrite, 0.0% in my test) and labels the RAG option with the 70% figure. RAG did improve judged accuracy (4.26 against 3.70) and retrieved the right category for 96% of products, so the retrieval works; it is the 3B model copying from the examples that fails. I stopped at four arms and did not tune further on the same 50 products, which would have overfit.

**24. How did you stop assumptions from being read as results?**
Every input in the impact section is labelled MEASURED, ASSUMPTION or ILLUSTRATIVE, a list names what the data cannot support (returns, conversion, shopper time saved, satisfaction), and machine time is not counted as hands-on time. The two numbers I could not reproduce (search timings) stay in the scorecard as DRIFT, not hidden.

### MLOps

**25. How is the work reproducible and tracked?**
Fixed seeds and config-driven settings; one command (`run_all`) logs every component's metrics to an MLflow experiment under a parent run tagged with the git commit; the scorecard stamps that commit. A drift table compared 75 numbers that earlier phases recorded: 68 match within the precision recorded, 5 are timings and 2 are flagged timing drift. Data and weights are gitignored, so reproducing them takes hours; the steps are in `docs/reproduce.md`.

**26. What is the Docker story?**
Multi-stage, non-root Dockerfiles for the API and app and a compose file with Ollama, mounting data and models. I should be direct: Docker is not installed on my laptop, so I have not built or run them, and the image sizes in the docs are estimates. I wrote down the one-time commands and the disk warning, and the first build is the real test.

**27. What is your testing and CI strategy?**
222 tests with tiny fixtures and mocks, so no data, weights or Ollama are needed: an end-to-end photo-to-listing test uses a real tiny classifier and a mocked LLM. Coverage is 75.5%, measured in a clean checkout, because that is what CI sees; training scripts, plotting code and the real CLIP encoder are deliberately not unit-tested (the scorecard covers them). GitHub Actions runs ruff, black and pytest with a 70% gate. The workflow passed locally in simulation, but I have not yet seen it run on GitHub.

### Scale

**28. What would you do with 10x the data?**
Photos: 42k became 420k. Embedding took about 2 hours on CPU, so about 20 hours: I would move to a GPU and incremental embedding. Exact search stays under the 50 ms budget until about 400k items, so 10x is where HNSW starts to matter. Training would need a stronger backbone and the group-aware split extended to CLIP duplicates. Reviews: 166,607 chunks would become about 1.7 million, which outgrows my in-house BM25 and a single Chroma collection.

**29. And at production scale?**
Replace the 3B CPU model with a faster or hosted one behind the same interface and re-measure hallucination and faithfulness with an independent judge, since cost, latency and privacy become decisions. Run quality thresholds per category (jewellery was flagged 46.6% and watches 14% for text by global thresholds). Add monitoring for calibration, flag rates and refusal rates, a review queue that feeds back better labels, and real tests for prompt injection through review text. Keep a human reviewing drafts: nothing here supports auto-publishing generated copy.

**30. What is the biggest weakness, and what would you do first?**
That everything was measured on clean 60x80 studio photos, while the problem is phone photos from small sellers. I would collect a few hundred real seller photos, relabel them with a person, and re-measure the classifier, the quality thresholds and the consistency check, before touching anything else.

## 4. Five "what went wrong and how I fixed it" stories

Each is taken from [decisions.md](decisions.md) and the git history. Structure: situation, what I did wrong or assumed, how I found out, the fix, what I changed in my process.

**1. The duplicate threshold that made the split impossible (Phase 1).**
*Situation:* I grouped near-duplicate photos by perceptual-hash distance 2 before splitting. *Problem:* at that distance the groups chained, and shoes of different products merged into one 2,859-image component, so a 70/15/15 split was impossible (I got 65.6/14.1/20.3). *Fix:* distances 0 and 1 gave identical groups, so I used 1 and added a test that no group spans splits. *Lesson:* check the actual split proportions, not just that the code ran; leakage control is only useful if the split is still valid.

**2. Retrieval made the model worse (Phase 4).**
*Situation:* the textbook RAG design for listings. *Problem:* the first results showed 70% of listings containing a material, claim or number that was not given, because the 3B model copied facts from the retrieved examples. The first run also had two bugs in my own metric (confidence percentages counted as invented numbers, common words counted as brands). *Fix:* fixed the metric and re-scored every arm, added a guardrail-triggered rewrite arm (0.0% without examples), made the safe pipeline the app default and labelled RAG with its 70% figure. *Lesson:* design an ablation that can show your idea losing, and report it; also stop iterating on the same 50 products, because the final design was tuned on them and the notebook says so.

**3. The strict citation parser (Phase 5).**
*Situation:* answers must cite reviews like `[R1]`. *Problem:* my first parser required exactly that form, but the model writes `[R1, R2]` and puts ids in several fields, so it refused 82% of answerable questions. *Fix:* parse citations from anywhere, strip only ids that were not shown, and turn an answer with no valid citation into a refusal. False refusals fell to 31.1% and invalid ids stayed at 0 of 99, with a test for each form. *Lesson:* a validator that is stricter than the model's output format turns into a silent quality bug; measure refusal rates, not just citation correctness.

**4. The measurements that were wrong, twice (Phase 7).**
*Situation:* the unified evaluation compared a re-run with earlier numbers. *Problem:* two harness bugs. The RAM sampler matched only `ollama` and missed `llama-server`, reporting 2.3 GB when the true peak was 5.1 GB (the model server holds 3.5 GB). And a text-search results file had a third "reference" row that overwrote the real key, so a metric looked like mAP 0.903 instead of 0.372. *Fix:* match both processes, key the rows properly, add a regression test. *Lesson:* a drift check is only as honest as its plumbing; I distrust a number that agrees too well, and the two real timing drifts (3x slower on battery) stayed in the scorecard instead of being re-measured until they looked right.

**5. The simple image checks that did not work (Phase 6).**
*Situation:* the brief asked for simple OpenCV checks for background clutter and text. *Problem:* a border-band measure flagged clean photos because models and products cross the border; at the 3% false-alarm cap its recall was only 0.9% to 2.6%, and the blob heuristic for text found about 1% of overlaid text. *Fix:* corner patches for clutter (recall 45.2% on test) and CLIP zero-shot for text (60.4% recall), keeping the heuristic as the baseline in the report. *Lesson:* write the failing baseline down; it justifies the heavier method. I also disclosed what is still weak: overexposure recall is 0.115, text precision is 0.387, and jewellery and watches are over-flagged by global thresholds.
