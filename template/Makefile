.PHONY: help init setup setup-prod test pipeline pipeline-prod train api inference promote data-pull data-push model-push model-pull deploy-model monitor up down reset clean all

# Docker compose shorthand
DC := docker compose

help:
	@echo "🚀 ML Pipeline Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make init              - Initialize local dev environment"
	@echo "  make setup             - Full dev setup (local + containers)"
	@echo "  make setup-prod        - Production setup"
	@echo ""
	@echo "Execution (all run inside Docker):"
	@echo "  make train             - Train model inside Docker"
	@echo "  make test              - Run tests inside Docker"
	@echo "  make pipeline          - Full DVC pipeline inside Docker"
	@echo "  make pipeline-prod     - Production pipeline with data pull"
	@echo ""
	@echo "Runtime:"
	@echo "  make api               - Start FastAPI server"
	@echo "  make inference         - Test inference endpoint"
	@echo ""
	@echo "Lifecycle:"
	@echo "  make up                - Start full system (MLflow + API + monitoring)"
	@echo "  make down              - Stop all containers"
	@echo "  make reset             - Stop containers and clean artifacts"
	@echo ""
	@echo "Utilities:"
	@echo "  make promote           - Run model promotion gate"
	@echo "  make monitor           - Run drift monitoring"
	@echo "  make data-pull         - Pull processed data from DVC remote"
	@echo "  make data-push         - Push processed data to DVC remote"
	@echo "  make model-pull        - Pull models from DVC remote"
	@echo "  make model-push        - Push models to DVC remote"

# ──────────────────────────────────────────────────────────────────────────
# SETUP & INITIALIZATION
# ──────────────────────────────────────────────────────────────────────────

init:
	poetry install --with dev
	git init || true

setup: init
	poetry run dvc init -q || true
	$(DC) up -d

# Production setup
setup-prod:
	poetry install --no-dev
	dvc init --no-scm
	@echo "✓ Configure your DVC remotes by editing .dvc/config"
	@echo "  S3: dvc remote add -d storage s3://your-bucket/dvc-cache"
	@echo "  PostgreSQL: dvc remote add -d storage postgresql://user:pass@localhost:5432/dvc_cache"

# S3-based production setup
init-prod-s3: setup-prod
	dvc remote add -d storage s3://your-production-bucket/dvc-cache || echo "✓ Storage remote configured"
	dvc remote add models s3://your-models-bucket/models || echo "✓ Models remote configured"
	dvc remote add data s3://your-data-bucket/data || echo "✓ Data remote configured"

# PostgreSQL-based production setup
init-prod-postgres: setup-prod
	@echo "Setting up PostgreSQL remotes..."
	dvc remote add -d storage postgresql://$${POSTGRES_USER:-user}:$${POSTGRES_PASSWORD:-password}@$${POSTGRES_HOST:-localhost}:$${POSTGRES_PORT:-5432}/dvc_cache || echo "✓ Storage remote configured"
	dvc remote add models postgresql://$${POSTGRES_USER:-user}:$${POSTGRES_PASSWORD:-password}@$${POSTGRES_HOST:-localhost}:$${POSTGRES_PORT:-5432}/models || echo "✓ Models remote configured"
	dvc remote add data postgresql://$${POSTGRES_USER:-user}:$${POSTGRES_PASSWORD:-password}@$${POSTGRES_HOST:-localhost}:$${POSTGRES_PORT:-5432}/data || echo "✓ Data remote configured"

init-prod: init-prod-s3

# ──────────────────────────────────────────────────────────────────────────
# DOCKER-FIRST EXECUTION (all core pipeline runs in containers)
# ──────────────────────────────────────────────────────────────────────────

# Train inside Docker (reproducible, MLflow-tracked)
train:
	$(DC) run --rm compute poetry run python -m {{project_name}}.src.models.train

# Test inside Docker (isolated environment)
test:
	$(DC) run --rm compute poetry run pytest {{project_name}}/tests/ -v --tb=short

# Full DVC pipeline inside Docker (deterministic, reproducible)
pipeline: data-pull
	$(DC) run --rm compute poetry run dvc repro

# Production pipeline with forced re-execution
pipeline-prod: data-pull
	$(DC) run --rm compute poetry run dvc repro --force

# Validate configuration inside Docker
validate-config:
	$(DC) run --rm compute poetry run python -m {{project_name}}.configs.validate

# ──────────────────────────────────────────────────────────────────────────
# RUNTIME SERVICES
# ──────────────────────────────────────────────────────────────────────────

# Start FastAPI server (long-running)
api:
	$(DC) up --build api

# Test inference against running API
inference:
	@echo "Testing inference endpoint..."
	curl -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"text": "this movie is great"}' \
		-w "\nStatus: %{http_code}\n"

# ──────────────────────────────────────────────────────────────────────────
# MODEL & DATA MANAGEMENT
# ──────────────────────────────────────────────────────────────────────────

# Promotion gate
promote:
	$(DC) run --rm compute poetry run python -m promotion.gate

# Data management
data-pull:
	dvc pull {{project_name}}/data/processed || echo "✓ No data to pull"

data-push:
	dvc push {{project_name}}/data/processed

model-pull:
	dvc pull {{project_name}}/models/artifacts

model-push:
	dvc push {{project_name}}/models/artifacts

# ──────────────────────────────────────────────────────────────────────────
# DEPLOYMENT & MONITORING
# ──────────────────────────────────────────────────────────────────────────

# Deploy model to MLflow registry
deploy-model: model-push
	$(DC) run --rm compute poetry run python -m {{project_name}}.serving.deploy

# Monitor model drift
monitor:
	$(DC) run --rm compute poetry run python -m {{project_name}}.monitoring.drift

# ──────────────────────────────────────────────────────────────────────────
# SYSTEM LIFECYCLE
# ──────────────────────────────────────────────────────────────────────────

# Start full system (MLflow + API + Prometheus + Grafana)
up:
	$(DC) up -d --build

# Stop all containers
down:
	$(DC) down

# Full reset (stop containers, clean volumes and cache)
reset: down
	$(DC) down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	dvc gc -w

# Quick development setup
all: init setup test pipeline

# ──────────────────────────────────────────────────────────────────────────
# CLEANUP
# ──────────────────────────────────────────────────────────────────────────

clean: reset
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".dvc" -prune -o -type d -name ".git" -prune || true
