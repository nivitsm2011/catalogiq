# Showcase text: resume and LinkedIn

Every number is from [reports/scorecard.md](../reports/scorecard.md) (section in brackets). Nothing here claims work that is not in this repository.
Before posting, check three things: the GitHub repository is **public**, the CI badge is green (the first run has not been confirmed), and the
link below is the one you want to share.

## Resume: 3 achievement bullets

Pick the order that matches the role. Each has an action, the technique and a measured result.

- **Trained and calibrated a multi-output MobileNetV3 product-attribute classifier** (7 attributes, near-duplicate-aware train/test split, class-weighted loss, temperature-scaled confidence thresholds) that reaches **0.771 mean macro-F1 on 6,339 held-out photos against 0.094 for a majority-class baseline**, at 13.9 ms per photo on CPU. [section 1]
- **Designed a guardrailed listing generator on a local Llama 3.2 3B** (hybrid BM25 + vector retrieval, deterministic fact-check, one-pass rewrite) and, through a 4-arm ablation on 50 held-out products, **found that retrieved examples raised the unsupported-fact rate from 2.0% to 70.0%; the guardrail rewrite brought the no-example pipeline to 0.0%**. [section 3]
- **Built a review Q&A system with citations** (product-scoped hybrid retrieval, evidence gate, citation validation) that reached **Recall@5 of 1.000 against 0.889 for vector-only retrieval, refused 93.3% of unanswerable questions and cited only review ids that exist (100%)**, and shipped it with a FastAPI backend, a Streamlit app and a one-command evaluation that reproduced 68 of 75 earlier numbers. [sections 4 and 7]

If a role wants one more line, these are also true and measured: CLIP photo search Recall@10 0.890 against 0.854 for the fine-tuned classifier's features [section 2]; an image and listing quality checker that flags blur with recall 0.999 and finds 47.7% of 300 injected photo-versus-listing mismatches [section 5].

Skills line (all used in this repo): PyTorch, transfer learning, model calibration, Grad-CAM, CLIP, FAISS, ChromaDB, RAG, Ollama, FastAPI, Streamlit, MLflow, pytest, Docker, GitHub Actions.

## LinkedIn post

> **A small language model will copy facts from the examples you give it. I measured how often.**
>
> Small online sellers upload products with poor photos, wrong categories and thin descriptions, which hurts search and returns. Shoppers can't get a quick answer about fit. So I built CatalogIQ, an AI copilot for marketplaces, end to end and entirely on a laptop CPU:
>
> - a photo becomes attributes (my own trained classifier), then a checked listing (a local 3B LLM with fact-check guardrails)
> - shoppers get search by words or photo, and answers that cite real reviews or admit there is not enough evidence
> - catalog teams get a quality score with plain-English fixes for every listing
>
> Three results from one evaluation scorecard:
> 1. The classifier reaches 0.771 mean macro-F1 on 6,339 held-out photos (a majority-class guess: 0.094), in about 14 ms per photo.
> 2. Giving the LLM example listings made 70% of its listings contain an unsupported fact, against 2% without examples. A deterministic fact-check plus one rewrite brought it to 0% in my 50-product test.
> 3. Review Q&A refused 93% of questions it should not answer and never cited a review that does not exist. The honest cost: it also refused 31% of questions it could have answered.
>
> What I learned: retrieval is not automatically better, a metric I cannot trust (an LLM judging itself) is worth less than a hand-check of 15 answers, and writing the limitations next to the results made the project more credible, not less. The photos are small studio thumbnails and the test sets are small, so this is a prototype, not a product claim.
>
> Code, scorecard and write-up: https://github.com/nivitsm2011/catalogiq
>
> #MachineLearning #RAG #ComputerVision #LLM #Python

## LinkedIn Featured section (short)

**Title:** CatalogIQ: an AI copilot for marketplaces (runs on a laptop CPU)

**Description:** Photo to checked listing, review Q&A with citations, and a catalog quality score. 0.771 mean macro-F1 on 6,339 held-out photos; retrieved examples made a 3B LLM invent facts in 70% of listings, and guardrails brought that to 0% in my test. Code, scorecard and honest limitations on GitHub.

**Link:** https://github.com/nivitsm2011/catalogiq
