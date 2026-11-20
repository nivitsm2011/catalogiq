# Architecture

CatalogIQ is five models and indexes behind one API, plus an evaluation harness. Everything runs locally on CPU; the only
network use is the one-time download of data and pretrained weights. Numbers quoted here come from [the scorecard](../reports/scorecard.md)
or from [the decisions log](decisions.md), where the reasoning and the alternatives are recorded in full.

## Component diagram

```mermaid
flowchart TB
    subgraph clients["Clients"]
        APP["Streamlit app<br/>app/Home.py + 4 pages"]
        DOCS["API docs /docs"]
    end
    subgraph backend["FastAPI backend (api/)"]
        MW["middleware: size cap, rate limit,<br/>request log + metrics"]
        RT["routes + pydantic schemas<br/>upload validation, one error shape"]
        SV["services.py"]
        ST["AppState: models loaded once,<br/>per-photo memo, warm-up thread"]
    end
    subgraph core["Python package (src/catalogiq/)"]
        V["vision<br/>MobileNetV3 multi-head<br/>calibration, TorchScript"]
        S["search<br/>CLIP, FAISS, shop-the-look,<br/>duplicates"]
        R["rag<br/>listing generator, guardrails,<br/>review Q&A, summaries"]
        Q["quality<br/>image checks, consistency,<br/>completeness, score"]
        L["llm<br/>LLMClient, OllamaClient,<br/>CachedClient"]
        E["eval<br/>run_all, drift check, scorecard"]
    end
    subgraph stores["Local stores (gitignored)"]
        D1[("data/processed<br/>parquet, embeddings")]
        D2[("ChromaDB<br/>listings + review chunks")]
        D3[("models/<br/>classifier, CLIP cache")]
        D4[("LLM reply cache")]
        D5[("MLflow file store")]
    end
    OLL["Ollama server<br/>llama3.2:3b"]
    APP --> MW --> RT --> SV --> ST
    DOCS --> MW
    ST --> V
    ST --> S
    ST --> R
    ST --> Q
    R --> L
    L --> D4
    L --> OLL
    V --> D3
    S --> D1
    S --> D3
    R --> D2
    E --> D5
    E -.re-runs.-> V
    E -.re-runs.-> S
    E -.re-runs.-> R
    E -.re-runs.-> Q
```

**Design rules that shape the code**
- **Config-driven.** Paths, seeds, thresholds, limits and prompts live in `configs/` ([reference](configuration.md)); nothing is hard-coded.
- **One interface for the language model.** Everything calls `LLMClient.chat`; Ollama is one implementation and a hosted model would be another. A disk cache keyed by the full request makes evaluations repeatable and the demo instant.
- **Models load once, at startup**, into one `AppState`; services take that state, so tests can pass fakes (222 tests run without any model).
- **Fail clearly.** The server checks its required files at startup, answers `503 llm_unavailable` when Ollama is down, and the app explains what still works.

## Data flow per feature

### 1. Snap-to-Listing: photo to attributes

```mermaid
flowchart LR
    A["Photo upload<br/>JPEG/PNG/WebP, 5 MB max"] --> B["Validate content,<br/>shrink to 1024 px"]
    B --> C["Resize 128x96,<br/>ImageNet normalise"]
    C --> D["MobileNetV3-Large backbone<br/>7 classification heads"]
    D --> E["Temperature scaling<br/>per attribute"]
    E --> F{"Confidence above<br/>the attribute threshold?"}
    F -- yes --> G["Pre-fill the value"]
    F -- no --> H["Show 'please confirm'<br/>with alternatives"]
```

- Training data: 42,257 cleaned products, split by near-duplicate group (pHash distance at most 1) so twins never straddle train and test.
- Imbalance: square-root inverse-frequency class weights; classes under 100 photos dropped or merged; colour collapsed to 15 families.
- Two stages on CPU: heads only (4 epochs), then fine-tuning (6 epochs).
- Calibration: one temperature per attribute fitted on validation; the threshold is the lowest confidence at which accepted validation predictions are at least 90% right.
- Result: mean macro-F1 0.771, 13.9 ms per photo (scorecard section 1).

### 2. Grounded listing copy

```mermaid
flowchart LR
    A["Confirmed attributes<br/>+ seller notes"] --> B["Category group<br/>(keyword rules)"]
    B --> C["Hybrid retrieval:<br/>BM25 + vector, RRF,<br/>quality boost"]
    B --> D["Category style rules"]
    C --> E["Prompt: facts, style rules,<br/>3 example listings"]
    D --> E
    E --> F["Llama 3.2 3B<br/>JSON schema output"]
    F --> G["Guardrails:<br/>unsupported facts, banned words,<br/>style limits"]
    G -- violations --> H["One rewrite<br/>with the problems listed"]
    H --> G2["Re-check, keep the better draft"]
    G -- clean --> I["Listing + fact-check report"]
    G2 --> I
```

- The knowledge base is 800 Amazon listings (a Bayesian-shrunk rating score picks exemplars) plus the style guide as 13 chunks, in ChromaDB.
- The hallucination check is deterministic: it compares the output with the attributes and seller notes only, using word lists in config. Retrieved examples are never a source of facts.
- Finding: with retrieved examples a 3B model copied facts from them, so 70.0% of listings contained an unsupported fact against 2.0% without examples; the rewrite brought the no-example arm to 0.0% (scorecard section 3). The app therefore defaults to the no-example pipeline plus rewrite and labels RAG with this warning.

### 3. Visual search, shop the look, duplicates

```mermaid
flowchart LR
    P["Photo or text query"] --> E["CLIP ViT-B/32<br/>512-d, L2-normalised"]
    E --> F{"Filters?"}
    F -- no --> X["FAISS IndexFlatIP<br/>exact search"]
    F -- yes --> Y["Restrict candidates,<br/>then exact search"]
    X --> R["Ranked catalog items"]
    Y --> R
    R --> L["Shop the look:<br/>role rules + score<br/>(pairing, colour, usage, season, CLIP)"]
    R --> D["Duplicate finder:<br/>CLIP cosine at least 0.975<br/>AND pHash distance at most 8"]
```

- All 42,257 catalog photos are embedded once (about 2 hours on CPU, chunked and resumable) and cached.
- Text queries are matched against image embeddings directly; the catalog-side text is an attribute sentence, never the product name.
- Results (scorecard section 2): Recall@10 0.890 for photo queries, 0.887 for text queries; vector search 23.0 ms and CLIP encoding 183 ms in the last run (both about 3x slower than first measured, flagged as drift, probably battery power).

### 4. Shopper Q&A over reviews

```mermaid
flowchart LR
    Q["Question about one product"] --> A["Retrieve inside that product:<br/>BM25 + vector chunks, RRF"]
    A --> B["Re-rank with helpful votes<br/>and verified purchase"]
    B --> G{"At least 2 reviews<br/>above similarity 0.52?"}
    G -- no --> R["Fixed refusal,<br/>no LLM call"]
    G -- yes --> L["Llama 3.2 3B:<br/>answer, agree, disagree,<br/>cited review ids"]
    L --> V["Parse and validate citations;<br/>no valid citation = refusal"]
    V --> O["Answer with citations<br/>or refusal"]
```

- 139,400 cleaned reviews become 166,607 chunks (whole if short, three-sentence windows if long).
- Results (scorecard section 4): Recall@5 1.000 and MRR 0.893 with the hybrid re-ranked retriever; 93.3% of unanswerable questions refused, 100% of cited ids exist; but 31.1% of answerable questions were refused too.
- The aspect summary (fit, fabric, colour, quality, comfort, delivery) computes every number in code; the LLM only phrases sentences, and a sentence with a number that is not in the facts is replaced by a template.

### 5. Catalog-quality checker

```mermaid
flowchart LR
    P["Photo + listing text"] --> I["Image checks:<br/>blur, exposure, resolution,<br/>aspect, corner clutter"]
    P --> T["CLIP zero-shot:<br/>text or watermark overlay"]
    P --> C["Consistency: classifier AND CLIP<br/>disagree with stated colour,<br/>type or gender"]
    L["Listing fields"] --> M["Completeness vs the<br/>category's required attributes"]
    I --> S["Score 0-100:<br/>40% image, 35% consistency,<br/>25% completeness"]
    T --> S
    C --> S
    M --> S
    S --> F["Flags with plain-English fixes<br/>below 85 = needs attention"]
```

- Thresholds were tuned on the validation split with a rule fixed beforehand (highest recall while flagging at most 3% of clean photos); test photos were never used for tuning.
- Defects were created by degrading clean photos, because the catalog is nearly defect-free (2 of 120 natural photos).
- Results (scorecard section 5): blur recall 0.999 / precision 0.423, too dark 1.000 / 0.899, mismatch precision 0.472 / recall 0.477; 8.3% of the 6,339 test listings carry a flag.

### 6. Evaluation and scorecard

`python -m catalogiq.eval.run_all` re-runs each component on its fixed test set, logs every metric to MLflow (experiment `catalogiq-eval`, tagged with the git commit), compares 75 numbers with what the earlier phases recorded, measures end-to-end latency, RAM and disk, and writes `reports/scorecard.md`. LLM steps replay through the reply cache with a budget of cache misses, so a changed cache stops with a message instead of silently running for hours.

## Why each technology (and what was rejected)

| Choice | Over | Why (most rows are recorded in [decisions.md](decisions.md); ChromaDB, FastAPI and Streamlit are general reasons, not measured comparisons) |
|---|---|---|
| MobileNetV3-Large, own fine-tuning | EfficientNet-B0, ResNet-50 | 5.4M parameters and a 13 MB export, built for CPU, 18.7 ms per photo measured at the time. EfficientNet-B0 was not benchmarked, so the choice rests on published size and measured MobileNet speed. |
| 128x96 network input | 224x224 | The source photos are about 60x80 pixels; 224 only adds interpolation and about 3x compute. |
| Class weights ~ (1/count)^0.5 | Oversampling, full inverse frequency | The data has near-duplicate photos, so duplicating rare images encourages memorising them. |
| Hue augmentation disabled (the code raises if set) | Standard colour jitter | A hue shift would turn a navy shirt purple and corrupt the colour label. |
| Group-aware split on pHash groups | Random split | The same photo in train and test inflates accuracy. At pHash distance 2 the groups chained into one 2,859-image cluster, so the threshold is 1. |
| Temperature scaling + one threshold per attribute | Platt or isotonic calibration | One number per attribute is explainable; the models were already well calibrated. |
| TorchScript export | ONNX | Output identical to the eager model; the migration path to ONNX or `torch.export` is noted because `torch.jit` is deprecated. |
| OpenCLIP ViT-B/32, zero-shot | Fine-tuned CLIP, the Phase 2 features | CLIP beat the fine-tuned classifier's features on Recall@10 (0.890 against 0.854) with no training. |
| FAISS exact index (HNSW optional) | HNSW by default | At 42k items exact search is a few milliseconds with perfect recall; the 50 ms budget is crossed at about 400,000 items. |
| ChromaDB | A hosted vector database | Local, file-based, no server; fits the one-folder, CPU-only constraint. |
| In-house BM25 (about 50 lines) + RRF | `rank_bm25`, weighted score fusion | No new dependency; RRF needs no score calibration. Hybrid reached Recall@5 1.000 against 0.889 for vectors alone. |
| all-MiniLM-L6-v2 embeddings | bge-small, nomic-embed-text | Smallest and fastest on CPU for 800 short documents. |
| Llama 3.2 3B via Ollama behind `LLMClient` | A hosted API, a 7B model | Runs on CPU and locally; the interface lets a hosted model replace it. A 7B model (deepseek-r1:7b) was considered for the judge and rejected as slow; it was not tried as the writer. |
| Deterministic guardrails | An LLM fact checker | Explainable and testable; the LLM judge was the same 3B model and agreed poorly with the owner. |
| Evidence gate before the LLM call | Letting the model decide to refuse | The gate refuses without spending 40 s of CPU; tuned by product-grouped cross-validation because a dev/test split did not generalise. |
| OpenCV single-number image checks | A learned defect classifier | Simple, fast, explainable; thresholds come from one rule. CLIP zero-shot replaced the text heuristic because the heuristic found about 1% of overlaid text. |
| FastAPI | Flask | Pydantic schemas, validation and OpenAPI docs come with it. |
| Streamlit | A JavaScript front end | One language and the fastest way to a four-page product-style demo; the API stays the real product surface. |
| MLflow local file store | SQLite store, a hosted tracker | The brief asked for a local file store; SQLite is the noted migration path. |
| pytest with fixtures and mocks, GitHub Actions | Tests that need real models | CI has no data or weights, so the 75.5% coverage figure is measured the way CI sees it. |

## Where the limits are

See [limitations and risks](limitations_and_risks.md) and the scorecard (section 9) for dataset bias, label noise, small-LLM limits and what would change at production scale.
