"""FastAPI entry point (placeholder until later phases)."""

from fastapi import FastAPI

app = FastAPI(title="CatalogIQ")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}
