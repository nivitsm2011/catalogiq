# CatalogIQ API

FastAPI service in `api/`. Interactive documentation (try every endpoint in the browser) is at **`/docs`** while the server runs.

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000     # or: .\tasks.ps1 api
.\.venv\Scripts\python.exe -m streamlit run app/Home.py            # or: .\tasks.ps1 app
```

All models are loaded **once at startup** (about 1.5-2 minutes on this CPU; `/health` reports `degraded` until they are ready and other
endpoints answer `503 starting_up`). Lazy parts (retriever, review texts, duplicate pairs, quality report) are warmed in a background thread.

## Endpoints

| Method and path | Input | What it returns |
|---|---|---|
| `POST /predict-attributes` | photo (multipart `file`) | 7 attributes with calibrated confidence, a `needs_review` flag, and runner-up values |
| `POST /generate-listing` | JSON: attributes (as confirmed or edited), `seller_notes`, `mode` (`no_rag_fix` default, `rag`, `no_rag`, `rag_fix`) | title, 5 bullets, description, tags, guardrail report (unsupported facts, banned words, style messages, coverage) |
| `POST /search/image` | photo, `k`, filters (`gender`, `baseColour`, `articleType`, `usage`, `season`, `masterCategory`), `exclude_id` | visually similar catalog items |
| `GET /search/text` | `q`, `k`, the same filters | items matching a description |
| `POST /shop-the-look` | form: `item_id` **or** a photo | complementary items by outfit role, with reasons |
| `POST /quality-check` | photo, optional `attributes` (JSON of stated values) and `title` | 0-100 score, components, flags each with a plain-English fix, photo metrics, mismatches, completeness |
| `POST /ask-reviews/{item_id}` | JSON `{"question": "..."}` | cited answer, confidence, agree / disagree lines, or "Not enough reviews mention this." |
| `GET /reviews/products` | | products with reviews; `demo=true` ones answer their sample questions instantly |
| `GET /reviews/{item_id}/summary` | `phrase` (bool) | aspect summary; numbers computed in code, sentence by template or LLM |
| `GET /ops/summary`, `GET /ops/duplicates`, `POST /ops/duplicates/decision` | | catalog quality KPIs and worst listings; duplicate review queue and decisions |
| `GET /health` | | status, uptime, loaded models, LLM reachability, model name, catalog size |
| `GET /metrics` | | request counts, errors, average and p95 latency per endpoint since start |
| `GET /demo/products`, `GET /meta/options`, `GET /catalog/image/{id}`, `GET /scorecard`, `GET /scorecard/metrics` | | demo data, dropdown values, catalog photo, evaluation scorecard |

## Validation and errors

- Photos: JPEG, PNG or WebP only, **at most 5 MB**, each side at least 16 px, at most 40 megapixels. The file's real content is checked, not just its declared type.
- Every error has the same shape and a readable message:

```json
{"error": {"code": "file_too_large", "message": "The photo is larger than the 5 MB limit."}}
```

| Status | Codes |
|---|---|
| 400 | `missing_file`, `empty_file`, `ambiguous_input`, `bad_content_length` |
| 404 | `unknown_item`, `unknown_product`, `no_scorecard`, `no_quality_report` |
| 413 | `file_too_large`, `image_too_large`, `request_too_large` (whole request above 5.5 MB) |
| 415 | `unsupported_media_type` |
| 422 | `invalid_image`, `image_too_small`, `validation_error` (lists the fields), `missing_attribute`, `unknown_attribute`, `unknown_filter`, `unknown_filter_value`, `invalid_attributes` |
| 429 | `rate_limited` (with a `Retry-After` header) |
| 502 | `llm_error` (the language model failed in an unexpected way) |
| 503 | `starting_up`, `llm_unavailable` (Ollama is not running or the model is not installed; the message says how to fix it) |
| 500 | `internal_error` (details only in the server log) |

## Limits (Phase 9, `configs/api.yaml`)

- **Request size:** a request whose `Content-Length` is above 5.5 MB is refused with `413 request_too_large` before the body is read.
- **Rate limit:** an in-process sliding window per client address: 120 requests per minute for every endpoint, 20 per minute for the endpoints that call the
  local LLM (`/generate-listing`, `/ask-reviews`, `/reviews`). `/health`, `/metrics`, `/docs` and catalog photos are exempt. Switch it off with
  `api.limits.rate_limit.enabled: false`. It is per process and in memory: a single-user safeguard, not a replacement for a gateway.
- **Ollama down:** `POST /generate-listing` and `POST /ask-reviews/{id}` answer `503 llm_unavailable` with instructions. Cached replies (the demo products) still work because they never reach Ollama.
- **Missing files:** at startup the server checks the files listed under `api.required_files` and exits with one line per missing file and the command that creates it.

## Latency (this laptop, CPU, measured against the running server)

Attributes about 1 s, text or image search about 0.4 s, outfit 0.7 s, quality check about 1 s, aspect summary about 1.3 s. **Listing generation and review
answers call the local LLM:** a new listing takes about 15 s (about a minute for the style-guided pipeline) and a new answer about 40 s. A reply that is
already in the LLM cache (all demo products) returns in about a second. Every request is logged with its latency and `/metrics` keeps the statistics.

## Design notes

- `api/state.py` loads the models once; `api/services.py` holds the logic (one function per endpoint) so it can be tested with fakes;
  `api/main.py` only wires routes, validation, error handling, request logging and metrics.
- LLM calls are serialised with a lock (one local model on one CPU).
- `tests/test_api.py` has 25 tests with FastAPI's `TestClient` and a fake application state (no models, no LLM): endpoints, validation of every upload rule,
  error shapes, metrics and the startup `503`.
