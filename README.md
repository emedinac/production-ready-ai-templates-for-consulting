# End-to-End ML Pipeline Template

Production-ready AI project template for consulting firms. Built with [Copier](https://copier.readthedocs.io) — supports updates back to existing projects.

## Overview

This template provides a standardized, production-grade foundation for AI/ML projects. It includes integrated tooling for data versioning, experiment tracking, model serving, and monitoring.

## Key Features

| Feature | Implementation |
|---------|----------------|
| **Data Versioning** | DVC pipeline (ingest → preprocess → train → eval) |
| **Experiment Tracking** | MLflow (via Docker Compose) |
| **Model Serving** | FastAPI with health checks & versioning |
| **Monitoring** | Prometheus + Grafana drift detection |
| **CI/CD** | GitHub Actions (test → train on PR → deploy on merge) |
| **Agentic Eval** | AGENTS.md eval loop |
| **Documentation** | ADR for every tool choice |

## Prerequisites

- Python 3.10+
- [Copier](https://copier.readthedocs.io) (`pip install copier`)
- Docker & Docker Compose (for MLflow, Prometheus, Grafana)

## Quick Start

### Generate a new project

```bash
# Generate a new project from the template
copier copy . /path/to/my_project
cd /path/to/my_project
```

### Update an existing project

```bash
cd my_project
copier update
```

## Project Structure

```
my_project/
├── configs/              # Configuration schemas & validation
├── promotion/           # Model promotion & rollback rules
├── registry/            # Model registry client
├── serving/             # FastAPI model serving
├── src/
│   ├── data/            # Data ingestion & preprocessing
│   ├── models/          # Training & evaluation
│   └── serving/         # API endpoints
├── tests/               # Test suite
├── dvc.yaml             # DVC pipeline definition
├── params.yaml          # Parameters configuration
└── docker-compose.yml   # Local dev services (MLflow, etc.)
```

## Available Make Targets

| Target | Description |
|--------|-------------|
| `make setup` | Install dependencies |
| `make test` | Run test suite |
| `make train` | Run training pipeline |
| `make serve` | Start FastAPI server |
| `make all` | Run full pipeline (setup → test → train → serve) |

## Documentation

- [ADR](./docs/adr/) — Architecture Decision Records
- [AGENTS.md](./AGENTS.md) — Agentic evaluation prompts
- [Monitoring](./docs/monitoring/) — Prometheus & Grafana configs

