# Testing

```powershell
.\.venv\Scripts\python.exe -m pytest                                  # all tests
.\.venv\Scripts\python.exe -m pytest --cov=src/catalogiq --cov=api    # with coverage (fails below 70%)
```

## What is tested, and how

| Area | Files | Approach |
|---|---|---|
| Data cleaning | `test_data.py`, `test_data_pipeline.py` | committed fixtures: a styles sample with a repaired and an unrecoverable line, photos for the image audit, review and metadata JSONL (streaming is replaced by the fixtures) |
| Prediction | `test_vision.py`, `test_modules_extra.py` | a tiny randomly initialised TorchScript network exported in the real layout (`tiny_model_dir`) |
| Search and duplicates | `test_search.py` | synthetic embeddings, flat and HNSW index agreement, filters, shop-the-look rules |
| Guardrails and RAG | `test_rag.py`, `test_qa.py`, `test_modules_extra.py` | a fake LLM with canned replies; unsupported claims, banned words and citations are checked |
| Quality checks | `test_quality.py`, `test_integration.py` | fixture photos with known defects (blurry, dark, cluttered, tiny) |
| Photo to listing, end to end | `test_integration.py` | real classifier + real prompt, guardrails and score; mocked LLM and CLIP |
| API | `test_api.py`, `test_api_ops.py`, `test_robustness.py`, `test_performance.py` | `TestClient` with a fake state; limits, Ollama-down and missing-file behaviour |
| Evaluation harness | `test_eval.py`, `test_eval_harness.py` | fake component runners, tmp output folders, no model |

Fixtures live in `tests/fixtures/` (about 18 KB, regenerate with `python tests/fixtures/make_fixtures.py`).
No test needs the real data, the trained model, Ollama or the network, so the same suite runs in GitHub Actions.

## Coverage

Measured with `pytest-cov` over `src/catalogiq` and `api`; the number is in the Phase 9 section of `docs/progress.md`.
Not covered on purpose: training and plotting scripts that need the full dataset and hours of CPU (`vision/train.py`,
`vision/baselines.py`, `vision/gradcam.py`, `search/viz.py`, `eval/components.py`, `rag/evaluate.py`, `rag/judge.py`,
`search/embeddings.py` for the real CLIP). Their outputs are checked by the evaluation harness (`reports/scorecard.md`), not by unit tests.
