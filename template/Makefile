.PHONY: help init setup setup-prod test pipeline train api inference promote \
        data-pull data-push model-push model-pull deploy-model monitor \
        up down reset clean all

DC := docker compose

# CONFIG
DATA_SOURCE ?= dvc   # options: dvc | local

ifeq ($(DATA_SOURCE),dvc)
DATA_CMD := $(DC) run --rm compute dvc pull $(PROJECT)/data/processed
else
DATA_CMD := echo "Using local data (no DVC pull)"
endif

# HELP
help:
	@echo "ML Pipeline"
	@echo "DATA_SOURCE=dvc|local make pipeline"

# SETUP
init:
	poetry install --with dev

setup: init
	$(DC) up -d --build

# CORE PIPELINE
data-preprocess:
	$(DC) run --rm compute python -m {{project_name}}.src.data.preprocess

pipeline:
	$(DATA_CMD)
	$(DC) run --rm compute python -m {{project_name}}.src.data.preprocess
	$(DC) run --rm compute python -m {{project_name}}.src.models.train

train:
	$(DC) run --rm compute python -m {{project_name}}.src.models.train

test:
	$(DC) run --rm compute pytest {{project_name}}/tests -v

# API
api:
	$(DC) up --build api

inference:
	curl -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"text": "this movie is great"}'

# DATA / MODEL
data-pull:
	$(DC) run --rm compute dvc pull {{project_name}}/data/processed || true

model-pull:
	$(DC) run --rm compute dvc pull {{project_name}}/models/artifacts

model-push:
	$(DC) run --rm compute dvc push {{project_name}}/models/artifacts

# ML LIFECYCLE
promote:
	$(DC) run --rm compute python -m promotion.gate

deploy-model: model-push
	$(DC) run --rm compute python -m {{project_name}}.serving.deploy

monitor:
	$(DC) run --rm compute python -m {{project_name}}.monitoring.drift

# SYSTEM
up:
	$(DC) up -d --build

down:
	$(DC) down

reset:
	$(DC) down -v

clean: reset

all: setup test pipeline

Modify preprocess so it obeys:

# if DATA_SOURCE=dvc -> load from /data/processed
# if DATA_SOURCE=hf -> download
# if DATA_SOURCE=local -> local files only