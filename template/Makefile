.PHONY: setup setup-prod validate-config test pipeline pipeline-prod train serve promote clean all deploy data-pull data-push model-push model-pull monitor init-prod

# Development setup
init:
	poetry run dvc init

setup:
	poetry install --with dev
	git init || true
	poetry run dvc init -q || true
	docker compose up -d

# Production setup
setup-prod:
	poetry install --no-dev
	dvc init --no-scm
	@echo "Configure your remotes by editing .dvc/config"
	@echo "Choose your storage backend:"
	@echo "  S3: dvc remote add -d storage s3://your-bucket/dvc-cache"
	@echo "  PostgreSQL: dvc remote add -d storage postgresql://user:pass@localhost:5432/dvc_cache"

# S3-based production setup
init-prod-s3: setup-prod
	dvc remote add -d storage s3://your-production-bucket/dvc-cache || echo "Configure storage remote"
	dvc remote add models s3://your-models-bucket/models || echo "Configure models remote"
	dvc remote add data s3://your-data-bucket/data || echo "Configure data remote"

# PostgreSQL-based production setup
init-prod-postgres: setup-prod
	@echo "Setting up PostgreSQL remotes..."
	dvc remote add -d storage postgresql://$${POSTGRES_USER:-user}:$${POSTGRES_PASSWORD:-password}@$${POSTGRES_HOST:-localhost}:$${POSTGRES_PORT:-5432}/dvc_cache || echo "Configure storage remote"
	dvc remote add models postgresql://$${POSTGRES_USER:-user}:$${POSTGRES_PASSWORD:-password}@$${POSTGRES_HOST:-localhost}:$${POSTGRES_PORT:-5432}/models || echo "Configure models remote"
	dvc remote add data postgresql://$${POSTGRES_USER:-user}:$${POSTGRES_PASSWORD:-password}@$${POSTGRES_HOST:-localhost}:$${POSTGRES_PORT:-5432}/data || echo "Configure data remote"

# Alias for backward compatibility
init-prod: init-prod-s3

validate-config:
	poetry run python -m {{project_name}}.configs.validate

test:
	poetry run pytest {{project_name}}/tests/ -v --tb=short

# Development pipeline
pipeline: validate-config
	poetry run dvc repro

# Production pipeline
pipeline-prod: validate-config data-pull
	poetry run dvc repro --force

train: validate-config
	poetry run python -m {{project_name}}.src.models.train

serve:
	poetry run uvicorn {{project_name}}.serving.api:app --reload --port 8000

promote:
	poetry run python -m promotion.gate

# Data management
data-pull:
	dvc pull {{project_name}}/data/processed || echo "No data to pull"

data-push:
	dvc push {{project_name}}/data/processed

# Model management
model-push:
	dvc push {{project_name}}/models/artifacts

model-pull:
	dvc pull {{project_name}}/models/artifacts

# Deployment
deploy-model: model-push
	poetry run python -m {{project_name}}.serving.deploy

# Monitoring
monitor:
	poetry run python -m {{project_name}}.monitoring.drift

all: setup test pipeline

clean:
	docker-compose down
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	dvc gc -w
