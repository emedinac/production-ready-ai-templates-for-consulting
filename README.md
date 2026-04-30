# AI Project Template

Production-ready AI project template for consulting firms.
Built with [Copier](https://copier.readthedocs.io) — supports updates back to existing projects.

## Generate a new project

```bash
pip install copier
copier copy gh:YOUR_USERNAME/ai-project-template my_project
cd my_project
make all
```

## Update an existing project when the template changes

```bash
cd my_project
copier update
```

This is the key advantage over Cookiecutter: template improvements
propagate to all client projects already using it.

## 'Initial ideas'
- [ ] DVC pipeline: ingest -> preprocess -> train -> eval
- [ ] MLflow tracking (Docker Compose)
- [ ] FastAPI serving with health checks and model versioning
- [ ] Prometheus + Grafana drift monitoring
- [ ] GitHub Actions CI (test -> train on PR -> deploy on merge)
- [ ] AGENTS.md agentic eval loop
- [ ] ADR documenting every tool choice
