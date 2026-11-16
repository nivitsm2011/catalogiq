# Running CatalogIQ with Docker

> **Status: written but not built.** Docker is not installed on the development laptop, so these
> files have not been built or run. They follow standard patterns and the YAML parses, but treat the
> first `docker compose build` as the real test and report any error. No image size below was
> measured; the figures are labelled **estimates**.

## What runs

| Service | Image | Port | Role |
|---|---|---|---|
| `ollama` | `ollama/ollama:latest` | 11434 | local language model server |
| `api` | built from `docker/Dockerfile.api` | 8000 | FastAPI backend (models load in about 2 minutes) |
| `app` | built from `docker/Dockerfile.app` | 8501 | Streamlit app (talks to the API only) |

Both Dockerfiles are multi-stage (a builder stage creates the virtualenv, the runtime stage copies it,
so compilers and pip caches never reach the final image), run as a non-root user (`catalogiq`, uid 10001)
and have a health check. The API image installs the CPU-only PyTorch wheel; the default Linux wheel would
add about 2 GB of CUDA libraries.

**Data and models are not inside the images.** `docker-compose.yml` mounts `./data` and `./models`
(read-write: saved duplicate decisions, cached CLIP weights) and `./reports` (read-only: scorecard and tuned
thresholds). Build them on the host first, following the README (data pipeline, classifier training,
embeddings, review index); the API refuses to start and lists exactly which files are missing otherwise.

## Disk usage: read this before building

The project folder is capped at about 5 GB (it is 4.75 GB today). Docker stores images and volumes **outside**
the project folder, inside Docker Desktop's virtual disk, so they add to the laptop's disk use but not to the
project footprint. Estimates, not measurements:

| Item | Estimate |
|---|---|
| `catalogiq-api` image (CPU PyTorch, open-clip, transformers, chromadb, faiss, opencv) | about 3 to 4 GB |
| `catalogiq-app` image (Streamlit, Plotly) | about 0.5 to 0.8 GB |
| `ollama/ollama` image | about 1.5 to 2 GB |
| `llama3.2:3b` model in the `ollama_models` volume | about 2 GB |
| Build cache left by the first build | up to a few GB |
| **Total** | **roughly 8 to 12 GB** |

If the laptop is short on disk, run only the app and the API on the host as described in the README (nothing
extra to store) and use Docker for a demo on another machine. After building you can reclaim the cache with
`docker builder prune`, and remove everything with `docker compose down --rmi all --volumes`.
The language model already stored by a host Ollama is **not** shared with the container: the compose
`ollama` service keeps its own copy in a volume (another 2 GB).

## Commands (Windows PowerShell)

Install Docker Desktop first (WSL 2 backend), then from the project root:

```powershell
docker compose build                                  # builds catalogiq-api and catalogiq-app
docker compose --profile setup run --rm ollama-pull   # ONE TIME: downloads llama3.2:3b (about 2 GB)
docker compose up                                     # api on :8000, app on :8501, ollama on :11434
```

Open `http://localhost:8501` for the app and `http://localhost:8000/docs` for the API. The API reports
`healthy` after about 2 to 4 minutes (model loading); the app waits for it.

```powershell
docker compose up -d                  # run in the background
docker compose logs -f api            # follow the API log
docker compose ps                     # health of each service
docker compose down                   # stop (volumes and images stay)
docker image ls catalogiq-api catalogiq-app   # real image sizes after the build
```

Build one image on its own:

```powershell
docker build -f docker/Dockerfile.api -t catalogiq-api .
docker build -f docker/Dockerfile.app -t catalogiq-app .
```

## Configuration

| Setting | Where | Default |
|---|---|---|
| Ollama address used by the API | env `CATALOGIQ_LLM_HOST` (set in compose) | `http://ollama:11434` |
| API address used by the app | env `CATALOGIQ_API_URL` (set in compose) | `http://api:8000` |
| Upload and rate limits | `configs/api.yaml` (baked into the image; edit and rebuild) | 5 MB photo, 120 requests per minute, 20 per minute for LLM endpoints |

## Known limits

- Both containers run on CPU. The first listing after starting is slow because Ollama loads the model.
- The image does not contain the data. Without the mounted `data/` and `models/` the API exits with a list of missing files.
- The compose file publishes the ports on all interfaces; bind them to `127.0.0.1:8000:8000` if the machine is on a shared network. The API has no authentication.
