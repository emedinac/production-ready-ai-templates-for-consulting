# ── Builder ─────────────────────────────────────────────
FROM python:3.14-slim AS builder

WORKDIR /app

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1

ARG POETRY_VERSION=2.3.2

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock ./

# Install deps only (no source yet → better caching)
RUN poetry install --only main --no-root

# ── Runtime ─────────────────────────────────────────────
FROM python:3.14-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi8 curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy installed deps from builder
COPY --from=builder /usr/local /usr/local

# Copy pyproject.toml for poetry commands
COPY pyproject.toml poetry.lock ./

# Copy source code
COPY {{project_name}} ./{{project_name}}

# Create non-root user
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# Default: run API server
# Can be overridden via docker-compose entrypoint
CMD ["uvicorn", "{{project_name}}.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]