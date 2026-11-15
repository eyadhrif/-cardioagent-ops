# Service 3: audio-engine - multi-stage build. One image, two uses (see docker-compose.yml):
#   - default command: the HTTP service POST /predict (ONNX Runtime + librosa), port 8001
#   - audio-trainer:   python -m audio_engine.train_svm (training, tracked in MLflow)
# Build from the repo root:  docker build -f docker/audio-engine.Dockerfile -t audio-engine .

# ---------- Stage 1: install the dependencies with uv ----------
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# No .python-version here: it says 3.13 (gateway), this image uses the 3.11 of the base image.
COPY pyproject.toml uv.lock ./
# Base dependencies + the "audio" group only (no tests, no notebook packages)
RUN uv sync --locked --no-default-groups --group audio

# ---------- Stage 2: final image ----------
FROM python:3.11-slim

# libsndfile1: C library used by soundfile/librosa to read .wav files
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN useradd --create-home appuser

COPY --from=builder /app/.venv /app/.venv
# The code AND the exported model audio_engine/cardio_model.onnx
COPY audio_engine/ ./audio_engine/

# The cache folder (features.csv) must be writable by the non-root user
RUN mkdir -p /app/cache && chown appuser:appuser /app/cache

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    DATA_DIR=/app/data \
    FEATURES_CSV=/app/cache/features.csv

USER appuser
# Port 8001 is used inside docker-network only (no "ports:" in docker-compose.yml)
EXPOSE 8001
CMD ["uvicorn", "audio_engine.service:app", "--host", "0.0.0.0", "--port", "8001"]
