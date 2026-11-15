# Service 5: api-gateway (FastAPI / Uvicorn) - multi-stage build
# Build from the repo root:  docker build -f docker/api-gateway.Dockerfile -t api-gateway .

# ---------- Stage 1: install the dependencies with uv ----------
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Copy only the dependency files first, so Docker can cache this layer
COPY pyproject.toml uv.lock ./
# --locked: use exactly the versions of uv.lock / --no-default-groups: no tests, no notebook packages
RUN uv sync --locked --no-default-groups

# ---------- Stage 2: small final image (no uv, no build cache) ----------
FROM python:3.13-slim

WORKDIR /app
RUN useradd --create-home appuser

COPY --from=builder /app/.venv /app/.venv
COPY api/ ./api/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
