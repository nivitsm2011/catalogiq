# Configuration reference

Nothing is hard-coded: paths, seeds, thresholds and limits live in `configs/*.yaml` and are read through
`src/catalogiq/config.py`. Every key below is read by code (unused keys were removed in Phase 9). Paths are
relative to the project root.

| File | Read by | What it controls |
|---|---|---|
| [`base.yaml`](../configs/base.yaml) | `get_config()` | project seed, folder layout, dataset locations, cleaning rules, CLIP and LLM settings, logging, MLflow |
| [`vision.yaml`](../configs/vision.yaml) | `get_vision_config()` | attribute classifier: image sizes, backbone, training, augmentation, calibration, inference files |
| [`search.yaml`](../configs/search.yaml) | `get_search_config()` | CLIP batch sizes, embedding files, index type, shop-the-look rules, duplicate detection |
| [`rag.yaml`](../configs/rag.yaml) | `get_rag_config()` | listing generator: embedder, retrieval, category rules, style, guardrails, LLM caching, seller-note keys |
| [`qa.yaml`](../configs/qa.yaml) | `get_qa_config()` | review Q&A: chunking, retrieval, aspect tagging, evidence thresholds, summaries |
| [`quality.yaml`](../configs/quality.yaml) | `get_quality_config()` | image checks, degradations, consistency check, completeness, score weights and fix texts |
| [`eval.yaml`](../configs/eval.yaml) | `get_eval_config()` | unified evaluation: output files, fixed test sets, drift tolerances, system run, impact assumptions |
| [`api.yaml`](../configs/api.yaml) | `get_api_config()` | API version, upload rules, request-size and rate limits, files required at startup |
| `colour_map.yaml` | `data.cleaning.load_colour_map` | raw colour name to retail colour family |
| `eval_reference.yaml` | `eval.reference` | the numbers recorded in earlier phases, used by the drift check |
| `qa_test_questions.yaml`, `retrieval_test_queries.yaml` | evaluations | the fixed question and query sets |
| `prompts/*.yaml` | generator, Q&A, judges | every LLM prompt (listing, Q&A, summary, judges, zero-shot aspect tagging) |

## Environment variables

| Variable | Effect | Default |
|---|---|---|
| `CATALOGIQ_LLM_HOST` | overrides `models.llm.host` (the Docker compose file sets it to the `ollama` container) | `http://localhost:11434` |
| `CATALOGIQ_API_URL` | address of the API used by the Streamlit app | `http://127.0.0.1:8000` |
| `MLFLOW_ALLOW_FILE_STORE` | set to `true` by the code; MLflow 3.16 needs an explicit opt-in for the local file store | `true` |

## `api.yaml` in detail (Phase 9)

| Key | Meaning | Default |
|---|---|---|
| `api.version` | reported by `/health` and the OpenAPI document | `0.9.0` |
| `api.upload.max_bytes` | largest accepted photo | 5 MB |
| `api.upload.max_pixels` | largest accepted photo in pixels (protects against decompression bombs) | 40 million |
| `api.upload.min_side` | smallest accepted side in pixels | 16 |
| `api.limits.max_body_bytes` | whole-request cap, checked from `Content-Length` before the body is read | 5.5 MB |
| `api.limits.rate_limit.enabled` | switch the in-process rate limiter off | `true` |
| `api.limits.rate_limit.window_s` | sliding-window length | 60 s |
| `api.limits.rate_limit.max_requests` | requests per client per window (all endpoints except the exempt ones) | 120 |
| `api.limits.rate_limit.heavy_max_requests` | requests per client per window to endpoints that call the local LLM | 20 |
| `api.limits.rate_limit.heavy_paths` / `exempt_paths` | path prefixes counted as heavy / never limited | see file |
| `api.required_files` | each file the server needs at startup and the command that creates it | see file |

## Settings worth knowing

- **LLM:** `models.llm.model` (default `llama3.2:3b`), `models.llm.fallback_model` (`qwen2.5:3b`, informational: put it in `model` to switch),
  `models.llm.timeout_s`, `models.llm.keep_alive` (how long Ollama keeps the model in memory; added in Phase 9, see
  [performance](../reports/phase9/performance.md)) and `rag.llm.cache` (the on-disk reply cache that makes the demo products instant).
- **Seeds:** `project.seed`, `rag.seed`, `qa.seed`, `quality.seed`, `eval.seed` make every run repeatable.
- **Quality thresholds:** `quality.image.thresholds` are defaults overridden by `reports/phase6/thresholds.json` after tuning.
- **Score:** `quality.score.weights` (image 40, consistency 35, completeness 25) and `quality.score.issue_below` (85).
