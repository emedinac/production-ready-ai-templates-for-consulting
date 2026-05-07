.PHONY: help init setup setup-prod test pipeline pipeline-prod train api inference promote data-pull data-push model-push model-pull deploy-model monitor up down reset clean all

DC := docker compose

# ──────────────────────────────────────────────────────────────────────────
# HELP
# ──────────────────────────────────────────────────────────────────────────

help:
	@echo "🚀 ML Pipeline (container-only)"

# ──────────────────────────────────────────────────────────────────────────
# SETUP (host bootstrap only, not runtime)
# ──────────────────────────────────────────────────────────────────────────

init:
	poetry install --with dev
	git init || true

setup: init
	$(DC) up -d --build

setup-prod:
	$(DC) up -d --build

# ──────────────────────────────────────────────────────────────────────────
# TRAIN / TEST / PIPELINE (ALL INSIDE CONTAINERS)
# ──────────────────────────────────────────────────────────────────────────

train:
	$(DC) run --rm compute python -m my_ai_project.src.models.train

test:
	$(DC) run --rm compute pytest my_ai_project/tests -v --tb=short

pipeline: data-pull
	$(DC) run --rm compute dvc repro

pipeline-prod: data-pull
	$(DC) run --rm compute dvc repro --force

validate-config:
	$(DC) run --rm compute python -m my_ai_project.configs.validate

# ──────────────────────────────────────────────────────────────────────────
# API (service mode)
# ──────────────────────────────────────────────────────────────────────────

api:
	$(DC) up --build api

inference:
	curl -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"text": "this movie is great"}'

# ──────────────────────────────────────────────────────────────────────────
# DVC / DATA / MODEL
# ──────────────────────────────────────────────────────────────────────────

data-pull:
	dvc pull my_ai_project/data/processed || true

data-push:
	dvc push my_ai_project/data/processed

model-pull:
	dvc pull my_ai_project/models/artifacts

model-push:
	dvc push my_ai_project/models/artifacts

# ──────────────────────────────────────────────────────────────────────────
# ML LIFECYCLE
# ──────────────────────────────────────────────────────────────────────────

promote:
	$(DC) run --rm compute python -m promotion.gate

deploy-model: model-push
	$(DC) run --rm compute python -m my_ai_project.serving.deploy

monitor:
	$(DC) run --rm compute python -m my_ai_project.monitoring.drift

# ──────────────────────────────────────────────────────────────────────────
# SYSTEM
# ──────────────────────────────────────────────────────────────────────────

up:
	$(DC) up -d --build

down:
	$(DC) down

reset: down
	$(DC) down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	dvc gc -w

all: init setup test pipeline

clean: reset
	find . -type f -name "*.pyc" -delete