# Production-Ready AI Project Template for Consulting

A prototpying-first AI delivery framework designed to accelerate new client onboarding and eliminate repeat infrastructure work.

## Executive Summary

Consultancies lose weeks on project setup when every new engagement rebuilds the same infrastructure. This template is built to reduce that overhead by providing a fully configured AI project scaffold with data versioning, model tracking, serving, monitoring, and deployment automation.

This repository is a project template, not a one-off model. It is built to be cloned, customized with a single configuration file, and put into production quickly.

## Why It Matters for Prototpying

- Reduce client ramp-up from weeks to days
- Standardize model evaluation across projects
- Make delivery repeatable and audit-ready
- Give consultants a proven production architecture from day one

## What This Template Delivers

- Cookiecutter-style project scaffold with `{{project_name}}` placeholders
- Configurable runtime and experiment behavior from a single `params.yaml`
- Preconfigured DVC pipeline for preprocess → train → evaluate
- Included MLflow tracking server in Docker Compose
- FastAPI inference service with health checks and Prometheus metrics
- Prometheus + Grafana observability with dashboard provisioning
- Standard pytest-based model and integration testing
- Docker Compose development and production profiles
- Makefile targets for setup, pipeline execution, deployment, and monitoring

## Supported Use Case

The reference implementation includes a text classification workflow. This is intended to illustrate a concrete, client-ready use case with a reusable pattern for ticket classification, support routing, or other text classification applications.

To adapt to a new client, you typically only need to update:

- `params.yaml`
- input data under `{{project_name}}/data/`

## Quick Start

### Bootstrap the Template

```bash
make setup
```

### Run the Full Delivery Flow

```bash
make all
```

This performs dependency installation, launches the application stack, runs tests, and executes the end-to-end pipeline.

Or
```bash
make setup
make pipeline
make api
make inference
```

### Run the Pipeline

```bash
make pipeline
```

### Run the API

```bash
make api
```

### Run Tests

```bash
make test
```

## Project Structure

```
{{project_name}}/
├── configs/           # Configuration loader and validation
├── data/              # Data tracked by DVC and used by the pipeline
│   ├── raw/           # Raw input data
│   └── processed/     # Processed artifacts
├── evaluation/        # Evaluation and reporting logic
├── experiments/       # Experiment orchestration and runners
├── models/            # Model artifacts and DVC-tracked outputs
├── monitoring/        # Drift detection and production monitoring
├── pipelines/         # DVC pipeline orchestration
├── serving/           # FastAPI inference and deployment helpers
├── src/               # Core application implementation
├── tests/             # Unit and integration tests
└── infra/             # Docker, monitoring, and deployment configuration
```

## Architecture Overview

- `docker-compose.yml`: local development stack with MLflow, PostgreSQL, compute, API, Prometheus, and Grafana
- `docker-compose.prod.yml`: production-ready service definitions and remote-aware storage
- `Makefile`: entrypoints for setup, pipeline execution, deployment, and monitoring
- `dvc.yaml`: staged data pipeline definition
- `params.yaml`: centralized configuration for model, data, and evaluation
- `{{project_name}}/serving/api.py`: FastAPI inference service with instrumentation

## Configuration

### Primary configuration file

`params.yaml` is the single source of truth for:

- task type (text classification, tabular regression, etc.)
- model architecture and hyperparameters
- data sources and preprocessing rules
- evaluation thresholds and metrics
- MLflow tracking settings

### Customizing for a new client

To onboard a new project, update `params.yaml` and provide the new dataset. The core pipeline and serving stack remain unchanged.

## Docker & Deployment

### Start local development stack

```bash
docker-compose up -d
```

### Start production stack

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Service endpoints

- API: http://localhost:8000
- MLflow UI: http://localhost:5000
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

## Observability

- MLflow captures experiment runs, parameters, metrics, and artifacts
- Prometheus scrapes application and inference metrics
- Grafana dashboards are shipped ready to provision from `infra/monitoring/grafana_dashboards`

## CI/CD

A GitHub Actions workflow is included for continuous validation of the DVC pipeline and project state.

The current pipeline automation covers:

- dependency installation
- DVC pull / repro / push lifecycle
- test execution

## Makefile Targets

- `make setup`: install dependencies and bring up the stack
- `make all`: full setup, test, and pipeline execution
- `make pipeline`: execute the preprocess + train flow
- `make test`: run pytest over the project tests
- `make api`: launch the FastAPI inference service
- `make monitor`: run drift detection and monitoring checks
- `make promote`: execute model promotion logic
- `make deploy-model`: push model artifacts and deploy the serving package

## Troubleshooting

### Common issues

- DVC remote connectivity
- MLflow backend availability
- PostgreSQL startup and database initialization
- API health endpoint failures

### Useful commands

```bash
docker-compose -f docker-compose.prod.yml ps db
psql postgresql://user:password@localhost:5432/dvc_cache -c "SELECT 1"
dvc status --show-json
```

## Business Value

This template is explicitly designed for prototpying teams that need to deliver AI projects quickly, consistently, and with minimal reprovisioning.

The key value proposition is:

- faster client onboarding
- repeatable delivery patterns
- standardized model evaluation
- ready-to-use production observability

