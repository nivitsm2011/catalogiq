# ruff: noqa: E501  (long endpoint descriptions and help strings)
"""CatalogIQ API: attributes from a photo, listings, search, outfits, quality checks, review Q&A.

Run:  python -m uvicorn api.main:app --port 8000      (interactive docs at /docs)
Models are loaded once at startup (about 30-60 s on CPU); every request is logged with its latency.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

import requests
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import services
from api.limits import make_limit_middleware
from api.metrics import RequestMetrics
from api.preflight import check_required_files
from api.schemas import (
    AskResponse,
    AspectSummaryResponse,
    DemoProduct,
    DuplicateDecision,
    GenerateListingRequest,
    GenerateListingResponse,
    HealthResponse,
    LookResponse,
    MetricsResponse,
    PredictAttributesResponse,
    QualityResponse,
    QuestionRequest,
    ReviewProduct,
    SearchResponse,
)
from api.state import AppState
from api.validation import ApiError, read_image
from catalogiq.config import get_api_config, get_config, resolve_path
from catalogiq.llm.client import LLMError, LLMUnavailable, is_unavailable_message
from catalogiq.logging_utils import get_logger

logger = get_logger("api")
TAGS = [
    {"name": "vision", "description": "Product attributes from a photo (Phase 2)."},
    {"name": "listing", "description": "Grounded listing copy with guardrails (Phase 4)."},
    {"name": "search", "description": "Visual and text search, outfits (Phase 3)."},
    {"name": "quality", "description": "Photo and listing quality score with fixes (Phase 6)."},
    {"name": "reviews", "description": "Shopper Q&A and aspect summaries over reviews (Phase 5)."},
    {"name": "ops", "description": "Catalog operations: quality report and duplicate queue."},
    {"name": "system", "description": "Health, metrics and reference data."},
]


def _error(status: int, code: str, message: str, details: Any = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


def llm_reachable() -> bool:
    """True when the Ollama server answers (it can stop on a sleeping laptop)."""
    host = get_config()["models"]["llm"]["host"]
    try:
        return requests.get(f"{host}/api/tags", timeout=1.5).ok
    except requests.RequestException:
        return False


def create_app(state: AppState | None = None, api_config: dict[str, Any] | None = None) -> FastAPI:
    """Build the app. Pass a ready ``AppState`` (tests use fakes) or let startup load the real one.

    ``api_config`` replaces ``configs/api.yaml`` (tests use it to set tiny limits).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        if state is None:
            check_required_files()  # a clear list of missing files instead of a stack trace
        app.state.catalogiq = state if state is not None else await run_in_threadpool(AppState.load)
        logger.info("startup complete: models loaded")
        if state is None:  # the real server: warm the lazy parts without delaying readiness
            threading.Thread(
                target=services.warm_up, args=(app.state.catalogiq,), daemon=True
            ).start()
        yield

    app = FastAPI(
        title="CatalogIQ API",
        version=get_api_config()["version"],
        description=(
            "AI copilot for e-commerce catalogs. Photos up to 5 MB (JPEG, PNG, WebP). "
            "Everything runs locally on CPU; listing generation and review answers call a local LLM "
            "and can take from seconds to about a minute unless the reply is cached."
        ),
        openapi_tags=TAGS,
        lifespan=lifespan,
    )
    metrics = RequestMetrics()
    app.state.metrics = metrics

    # ---------------------------------------------------------------- middleware + errors
    app.middleware("http")(
        make_limit_middleware(api_config)
    )  # inner: rejected requests are still logged

    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            ms = 1000 * (time.perf_counter() - start)
            route = request.scope.get("route")
            name = f"{request.method} {route.path if route else request.url.path}"
            metrics.record(name, status, ms)
            logger.info("%s %s -> %d in %.0f ms", request.method, request.url.path, status, ms)

    @app.exception_handler(LLMError)
    async def llm_error(_: Request, exc: LLMError) -> JSONResponse:
        down = isinstance(exc, LLMUnavailable) or is_unavailable_message(str(exc))
        return _error(503 if down else 502, "llm_unavailable" if down else "llm_error", str(exc))

    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {
                "field": ".".join(str(p) for p in e["loc"][1:]) or str(e["loc"][0]),
                "message": e["msg"],
            }
            for e in exc.errors()
        ]
        return _error(
            422,
            "validation_error",
            "The request is not valid: "
            + "; ".join(f"{d['field']}: {d['message']}" for d in details),
            details,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        return _error(
            500,
            "internal_error",
            f"Unexpected error ({exc.__class__.__name__}). See the server log.",
        )

    def st(request: Request) -> AppState:
        s = getattr(request.app.state, "catalogiq", None)
        if s is None:
            raise ApiError(
                503, "starting_up", "The models are still loading; try again in a minute."
            )
        return s  # type: ignore[no-any-return]

    # --------------------------------------------------------------------------- system
    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health(request: Request) -> HealthResponse:
        """Liveness and readiness: which models are loaded and whether the LLM server answers."""
        s = getattr(request.app.state, "catalogiq", None)
        if s is None:
            return HealthResponse(
                status="degraded",
                uptime_s=time.time() - metrics.started,
                models={},
                llm_reachable=False,
                llm_model=get_config()["models"]["llm"]["model"],
                version=get_api_config()["version"],
                catalog_size=0,
            )
        reachable = await run_in_threadpool(llm_reachable)
        ok = bool(s.models) and all(s.models.values())
        return HealthResponse(
            status="ok" if ok else "degraded",
            uptime_s=time.time() - s.started,
            models=s.models,
            llm_reachable=reachable,
            llm_model=getattr(s.llm, "model", "unknown"),
            version=s.version,
            catalog_size=int(len(s.catalog)),
        )

    @app.get("/metrics", response_model=MetricsResponse, tags=["system"])
    async def get_metrics() -> dict[str, Any]:
        """Request counts and average / p95 latency per endpoint since the server started."""
        return metrics.snapshot()

    @app.get("/meta/options", tags=["system"])
    async def options(request: Request) -> dict[str, Any]:
        """Attribute values for dropdowns and the values each search filter accepts."""
        s = st(request)
        return {
            "attribute_labels": s.labels,
            "filters": {
                c: sorted(s.catalog[c].dropna().astype(str).unique().tolist())
                for c in services.FILTER_COLUMNS
            },
        }

    @app.get("/scorecard", response_class=PlainTextResponse, tags=["system"])
    async def scorecard() -> str:
        """The evaluation scorecard (markdown) written by `python -m catalogiq.eval.run_all`."""
        path = resolve_path("reports/scorecard.md")
        if not path.exists():
            raise ApiError(
                404,
                "no_scorecard",
                "Run `python -m catalogiq.eval.run_all` to create the scorecard.",
            )
        return path.read_text(encoding="utf-8")

    @app.get("/scorecard/metrics", tags=["system"])
    async def scorecard_metrics() -> dict[str, Any]:
        """Flat headline metrics of the last evaluation run (from reports/phase7/eval_results.json)."""
        path = resolve_path("reports/phase7/eval_results.json")
        if not path.exists():
            raise ApiError(404, "no_scorecard", "Run `python -m catalogiq.eval.run_all` first.")
        data = json.loads(path.read_text(encoding="utf-8"))
        flat: dict[str, Any] = {}
        for comp in data["components"].values():
            flat.update({k: v for k, v in comp["metrics"].items() if v is not None})
        drift = [r["status"] for r in data.get("drift", [])]
        return {
            "metrics": flat,
            "git": data.get("git", {}),
            "generated": data.get("generated"),
            "drift": {s: drift.count(s) for s in set(drift)},
        }

    @app.get("/catalog/image/{item_id}", tags=["search"], response_class=FileResponse)
    async def catalog_image(item_id: int) -> FileResponse:
        """The catalog photo of an item."""
        return FileResponse(
            services.catalog_image_path(item_id),
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=86400"},
        )

    @app.get("/demo/products", response_model=list[DemoProduct], tags=["system"])
    async def demo_products(request: Request) -> list[dict[str, Any]]:
        """Ten preloaded sample products (photo, attributes and seller notes) for demo mode."""
        return [{**d, "image_url": f"/catalog/image/{d['id']}"} for d in st(request).demo]

    # ---------------------------------------------------------------------------- vision
    @app.post("/predict-attributes", response_model=PredictAttributesResponse, tags=["vision"])
    async def predict_attributes(
        request: Request,
        file: Annotated[
            UploadFile | None, File(description="Product photo, JPEG/PNG/WebP, max 5 MB")
        ] = None,
    ) -> dict[str, Any]:
        """Predict masterCategory, subCategory, articleType, baseColour, gender, season and usage."""
        image = await read_image(file)
        return await run_in_threadpool(services.predict_attributes, st(request), image)

    # --------------------------------------------------------------------------- listing
    @app.post("/generate-listing", response_model=GenerateListingResponse, tags=["listing"])
    async def generate_listing(request: Request, body: GenerateListingRequest) -> dict[str, Any]:
        """Write title, 5 bullets, description and tags from confirmed attributes; guardrails check the result."""
        attrs = {k: v.model_dump() for k, v in body.attributes.items()}
        return await run_in_threadpool(
            services.generate_listing, st(request), attrs, body.seller_notes, body.mode
        )

    # ---------------------------------------------------------------------------- search
    @app.post("/search/image", response_model=SearchResponse, tags=["search"])
    async def search_image(
        request: Request,
        file: Annotated[UploadFile | None, File()] = None,
        k: Annotated[int, Form(ge=1, le=50)] = 10,
        gender: Annotated[str | None, Form()] = None,
        baseColour: Annotated[str | None, Form()] = None,
        articleType: Annotated[str | None, Form()] = None,
        usage: Annotated[str | None, Form()] = None,
        season: Annotated[str | None, Form()] = None,
        masterCategory: Annotated[str | None, Form()] = None,
        exclude_id: Annotated[int | None, Form()] = None,
    ) -> dict[str, Any]:
        """Catalog items that look like the photo (CLIP), optionally filtered by attributes."""
        image = await read_image(file)
        s = st(request)
        filters = services.check_filters(
            s,
            {
                "gender": gender,
                "baseColour": baseColour,
                "articleType": articleType,
                "usage": usage,
                "season": season,
                "masterCategory": masterCategory,
            },
        )
        return await run_in_threadpool(services.search_image, s, image, k, filters, exclude_id)

    @app.get("/search/text", response_model=SearchResponse, tags=["search"])
    async def search_text(
        request: Request,
        q: Annotated[
            str,
            Query(
                min_length=2, max_length=200, description="For example: black formal shoes for men"
            ),
        ],
        k: Annotated[int, Query(ge=1, le=50)] = 10,
        gender: str | None = None,
        baseColour: str | None = None,
        articleType: str | None = None,
        usage: str | None = None,
        season: str | None = None,
        masterCategory: str | None = None,
    ) -> dict[str, Any]:
        """Catalog items matching a text description (CLIP text to image), optionally filtered."""
        s = st(request)
        filters = services.check_filters(
            s,
            {
                "gender": gender,
                "baseColour": baseColour,
                "articleType": articleType,
                "usage": usage,
                "season": season,
                "masterCategory": masterCategory,
            },
        )
        return await run_in_threadpool(services.search_text, s, q, k, filters)

    @app.post("/shop-the-look", response_model=LookResponse, tags=["search"])
    async def shop_the_look(
        request: Request,
        item_id: Annotated[int | None, Form()] = None,
        file: Annotated[UploadFile | None, File()] = None,
    ) -> dict[str, Any]:
        """Complementary items (shoes for a shirt, a bag for a dress) for a catalog item id or a photo."""
        image = await read_image(file) if file is not None else None
        return await run_in_threadpool(services.shop_the_look, st(request), item_id, image)

    # --------------------------------------------------------------------------- quality
    @app.post("/quality-check", response_model=QualityResponse, tags=["quality"])
    async def quality_check(
        request: Request,
        file: Annotated[UploadFile | None, File()] = None,
        attributes: Annotated[
            str | None,
            Form(
                description='JSON of stated values, e.g. {"baseColour": "Blue", "articleType": "Shirts", "gender": "Men"}'
            ),
        ] = None,
        title: Annotated[str | None, Form(max_length=300)] = None,
    ) -> dict[str, Any]:
        """Photo checks (blur, light, background, text), photo-vs-listing consistency, completeness and score."""
        image = await read_image(file)
        stated: dict[str, str] = {}
        if attributes:
            try:
                parsed = json.loads(attributes)
                if not isinstance(parsed, dict) or not all(
                    isinstance(v, str) for v in parsed.values()
                ):
                    raise ValueError
                stated = parsed
            except ValueError as exc:
                raise ApiError(
                    422, "invalid_attributes", "'attributes' must be a JSON object of text values."
                ) from exc
        return await run_in_threadpool(services.quality_check, st(request), image, stated, title)

    # --------------------------------------------------------------------------- reviews
    @app.get("/reviews/products", response_model=list[ReviewProduct], tags=["reviews"])
    async def reviews_products(request: Request) -> list[dict[str, Any]]:
        """Reviewed products you can ask about; `demo` ones have instant answers for their sample questions."""
        return await run_in_threadpool(services.review_products, st(request))

    @app.get("/reviews/{item_id}/summary", response_model=AspectSummaryResponse, tags=["reviews"])
    async def reviews_summary(
        request: Request, item_id: str, phrase: bool = False
    ) -> dict[str, Any]:
        """Aspect summary (fit, fabric, colour, quality, comfort, delivery); numbers are computed, not generated."""
        return await run_in_threadpool(services.review_summary, st(request), item_id, phrase)

    @app.post("/ask-reviews/{item_id}", response_model=AskResponse, tags=["reviews"])
    async def ask_reviews(request: Request, item_id: str, body: QuestionRequest) -> dict[str, Any]:
        """Answer from this product's reviews only, with citations, or say there is not enough evidence."""
        return await run_in_threadpool(services.ask_reviews, st(request), item_id, body.question)

    # ------------------------------------------------------------------------------- ops
    @app.get("/ops/summary", tags=["ops"])
    async def ops_summary(
        request: Request,
        category: str | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> dict[str, Any]:
        """Quality KPIs, issues by type and category, and the worst listings with fix suggestions."""
        return await run_in_threadpool(services.ops_summary, st(request), category, limit)

    @app.get("/ops/duplicates", tags=["ops"])
    async def ops_duplicates(
        request: Request, limit: Annotated[int, Query(ge=1, le=50)] = 12
    ) -> dict[str, Any]:
        """The duplicate review queue: pairs flagged by CLIP similarity and pHash, not yet decided."""
        return await run_in_threadpool(services.duplicate_queue, st(request), limit)

    @app.post("/ops/duplicates/decision", tags=["ops"])
    async def ops_duplicate_decision(body: DuplicateDecision) -> dict[str, Any]:
        """Record whether a flagged pair is merged or kept as two listings."""
        return await run_in_threadpool(
            services.save_duplicate_decision, body.id_a, body.id_b, body.decision
        )

    return app


app = create_app()
