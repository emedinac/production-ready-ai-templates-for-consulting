FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    POETRY_VIRTUALENVS_CREATE=false

ARG POETRY_VERSION=2.3.2

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libffi-dev curl git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock ./

RUN poetry install --with dev --no-root --no-interaction --no-ansi

COPY . /app

RUN useradd -m appuser
USER appuser

CMD ["python", "-m", "{{project_name}}.src.models.train"]
