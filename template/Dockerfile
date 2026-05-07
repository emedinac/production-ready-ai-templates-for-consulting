# ── Builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

ENV POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

ARG POETRY_VERSION=2.3.2

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock ./

# IMPORTANT: install ALL runtime deps (not dev-only split issues)
RUN poetry install --only main --no-root --no-interaction --no-ansi


# ── Runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi8 curl git \
    && rm -rf /var/lib/apt/lists/*

# copy installed site-packages
COPY --from=builder /usr/local /usr/local

COPY {{project_name}} /app/{{project_name}}

RUN useradd -m appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "{{project_name}}.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]