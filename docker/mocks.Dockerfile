# Image for the MOCK services (fake audio-engine and fake agent-orchestrator).
# The same image is used twice in docker-compose.yml with a different "command".

FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-default-groups

COPY mocks/ ./mocks/
