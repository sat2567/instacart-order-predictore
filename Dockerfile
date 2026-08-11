# Multi-stage build:
#   builder  -> installs Python deps into an isolated venv (has compilers/cache)
#   runtime  -> slim image + venv + code, runs as a non-root user
#   pipeline / api -> same image, different default command

FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.12-slim AS runtime
# libgomp: OpenMP runtime needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PROJECT_ROOT=/app \
    ARTIFACT_DIR=/app/artifacts \
    DATA_DIR=/app/data
WORKDIR /app
COPY --chown=app:app sql ./sql
COPY --chown=app:app src ./src
RUN mkdir -p /app/artifacts /app/data && chown -R app:app /app
USER app

FROM runtime AS pipeline
CMD ["python", "-m", "reorder.pipeline", "all"]

FROM runtime AS api
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "reorder.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
