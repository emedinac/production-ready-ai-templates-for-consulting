.PHONY: setup validate-config test pipeline train serve promote clean all

setup:
	poetry install
	docker-compose up -d

validate-config:
	poetry run python -m {{project_name}}.configs.validate

test:
	poetry run pytest {{project_name}}/tests/ -v --tb=short

pipeline: validate-config
	poetry run dvc repro

train: validate-config
	poetry run python -m {{project_name}}.src.models.train

serve:
	poetry run uvicorn {{project_name}}.serving.api:app --reload --port 8000

promote:
	poetry run python -m promotion.gate

all: setup test pipeline

clean:
	docker-compose down
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	dvc gc -w
