# ruff: noqa: E501  (long endpoint descriptions and help strings)
"""Pydantic request and response models of the CatalogIQ API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------------- attributes
class Alternative(BaseModel):
    """A runner-up value for an attribute and its probability."""

    label: str
    probability: float


class AttributePrediction(BaseModel):
    """One predicted attribute with its calibrated confidence."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool = Field(
        description="True when the confidence is below the attribute's threshold"
    )
    threshold: float
    alternatives: list[Alternative] = Field(default_factory=list)


class ImageInfo(BaseModel):
    """Size of the uploaded photo."""

    width: int
    height: int


class PredictAttributesResponse(BaseModel):
    """Attributes predicted from a photo."""

    attributes: dict[str, AttributePrediction]
    image: ImageInfo
    latency_ms: float


class AttributeInput(BaseModel):
    """An attribute value as confirmed or edited by the seller."""

    label: str = Field(min_length=1, max_length=80)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    needs_review: bool = False


# --------------------------------------------------------------------------------- listing
class GenerateListingRequest(BaseModel):
    """Attributes (possibly edited by the seller), optional seller notes and the pipeline to use."""

    attributes: dict[str, AttributeInput]
    seller_notes: dict[str, str] = Field(default_factory=dict)
    mode: Literal["no_rag", "rag", "no_rag_fix", "rag_fix"] = Field(
        default="no_rag_fix",
        description="no_rag_fix = safest in the Phase 4 evaluation; rag = style-guided but copies facts",
    )


class ListingOut(BaseModel):
    """Generated listing copy."""

    title: str
    bullets: list[str]
    description: str
    search_tags: list[str]
    attributes_used: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Unsupported(BaseModel):
    """A fact the listing states that the input does not support."""

    kind: str
    term: str


class GuardrailOut(BaseModel):
    """Result of the deterministic guardrail checks."""

    hallucinated: bool
    unsupported: list[Unsupported]
    banned: list[str]
    style_violations: list[str]
    style_messages: list[str] = Field(default_factory=list)
    style_compliant: bool
    coverage: float
    missing_attributes: list[str]


class GenerateListingResponse(BaseModel):
    """A generated listing with its guardrail report."""

    listing: ListingOut | None
    guardrails: GuardrailOut | None
    group: str
    mode: str
    attempts: int
    fixed: bool
    retrieved: list[str]
    error: str | None = None
    elapsed_s: float = Field(description="Wall-clock time of this call (a cached reply is instant)")
    generation_s: float = Field(description="Time the model originally needed for this reply")


# ---------------------------------------------------------------------------------- search
class ItemCard(BaseModel):
    """A catalog item as shown in result lists."""

    id: int
    title: str
    articleType: str | None = None
    baseColour: str | None = None
    gender: str | None = None
    usage: str | None = None
    season: str | None = None
    masterCategory: str | None = None
    score: float | None = None
    image_url: str
    role: str | None = None
    reasons: str | None = None


class SearchResponse(BaseModel):
    """Search results."""

    query: str
    k: int
    filters: dict[str, str]
    results: list[ItemCard]
    latency_ms: float


class LookResponse(BaseModel):
    """Complementary items for an outfit."""

    query: dict[str, Any]
    items: list[ItemCard]
    note: str | None = None
    latency_ms: float


# --------------------------------------------------------------------------------- quality
class FlagOut(BaseModel):
    """One issue with a plain-English fix."""

    code: str
    area: str
    message: str
    fix: str
    penalty: float


class MismatchOut(BaseModel):
    """A stated attribute the photo does not seem to match."""

    attribute: str
    stated: str
    seen: str


class CompletenessOut(BaseModel):
    """Share of the category's required attributes that are present."""

    group: str
    required: list[str]
    missing: list[str]
    pct: float


class QualityResponse(BaseModel):
    """The listing quality score with flags and fixes."""

    score: float
    components: dict[str, float]
    needs_attention: bool
    flags: list[FlagOut]
    image_issues: list[str]
    image_metrics: dict[str, float]
    mismatches: list[MismatchOut]
    completeness: CompletenessOut
    latency_ms: float


# ------------------------------------------------------------------------------------ reviews
class QuestionRequest(BaseModel):
    """A shopper question about one product."""

    question: str = Field(min_length=3, max_length=500)


class CitationOut(BaseModel):
    """A review quoted in an answer."""

    review_id: str
    rating: float | None = None
    verified_purchase: bool | None = None
    helpful_vote: int | None = None
    date: str | None = None
    text: str


class AskResponse(BaseModel):
    """Answer to a shopper question, grounded in the product's reviews."""

    answer: str
    refused: bool
    confidence: float
    aspect: str
    evidence_count: int
    citations: list[CitationOut]
    reviewers_agree: str = ""
    reviewers_disagree: str = ""
    top_similarity: float
    elapsed_s: float
    invalid_citations: list[str] = Field(default_factory=list)
    error: str | None = None


class ReviewProduct(BaseModel):
    """A product with reviews that can be questioned."""

    item_id: str
    title: str
    store: str | None = None
    average_rating: float | None = None
    n_reviews: int
    demo: bool = Field(
        description="True when its demo questions are answered instantly from the LLM cache"
    )
    suggested_questions: list[str] = Field(default_factory=list)


class AspectSummaryResponse(BaseModel):
    """Aspect-level summary: numbers are computed in code, the LLM only phrases them."""

    item_id: str
    title: str
    n_reviews: int
    aspects: dict[str, dict[str, Any]]
    labels: dict[str, str]
    summary: str
    source: str


# ---------------------------------------------------------------------------------- demo + ops
class DemoProduct(BaseModel):
    """A preloaded sample product for demo mode."""

    id: int
    title: str
    group: str
    attributes: dict[str, AttributeInput]
    seller_notes: dict[str, str]
    image_url: str


class DuplicateDecision(BaseModel):
    """A catalog team decision about a flagged duplicate pair."""

    id_a: int
    id_b: int
    decision: Literal["merge", "keep_both"]


# -------------------------------------------------------------------------------- health etc.
class HealthResponse(BaseModel):
    """Liveness and readiness."""

    status: Literal["ok", "degraded"]
    uptime_s: float
    models: dict[str, bool]
    llm_reachable: bool
    llm_model: str
    version: str
    catalog_size: int


class EndpointMetrics(BaseModel):
    """Request statistics of one endpoint."""

    count: int
    errors: int
    avg_ms: float
    p95_ms: float


class MetricsResponse(BaseModel):
    """Request counts and latencies since the server started."""

    total_requests: int
    total_errors: int
    avg_latency_ms: float
    uptime_s: float
    endpoints: dict[str, EndpointMetrics]
