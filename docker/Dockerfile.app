# CatalogIQ Streamlit app image: small, multi-stage, non-root. It only calls the API over HTTP.
#
# Build (from the project root):  docker build -f docker/Dockerfile.app -t catalogiq-app .

# ------------------------------------------------------------------ stage 1: build the virtualenv
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY docker/requirements-app.txt /tmp/requirements-app.txt
RUN pip install -r /tmp/requirements-app.txt
RUN find /opt/venv -type d \( -name "__pycache__" -o -name "tests" -o -name "test" \) -prune -exec rm -rf {} + \
    && find /opt/venv -name "*.pyc" -delete

# ------------------------------------------------------------------ stage 2: runtime
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="catalogiq-app" \
      org.opencontainers.image.description="CatalogIQ Streamlit app"

RUN groupadd --gid 10001 catalogiq \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin catalogiq

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src:/app \
    CATALOGIQ_API_URL=http://api:8000 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
# The app imports a few small modules from src/catalogiq (config, logging, the LLM error messages).
COPY --chown=catalogiq:catalogiq src ./src
COPY --chown=catalogiq:catalogiq app ./app
COPY --chown=catalogiq:catalogiq configs ./configs
COPY --chown=catalogiq:catalogiq .streamlit ./.streamlit
RUN mkdir -p logs reports && chown -R catalogiq:catalogiq /app

USER catalogiq
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

CMD ["python", "-m", "streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
