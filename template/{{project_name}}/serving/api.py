from contextlib import asynccontextmanager
import logging

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app

from CHANGE_ME.serving.schema import PredictRequest
from CHANGE_ME.configs.validate import load_and_validate_config
from CHANGE_ME.serving.loader import load_latest_from_experiment


logger = logging.getLogger("api")

# Metrics
PREDICTION_COUNTER = Counter(
    "inference_requests_total",
    "Total number of inference requests",
)

PREDICTION_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
)


def ensure_experiment(name: str):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        mlflow.create_experiment(name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_and_validate_config()

    mlflow.set_tracking_uri("http://mlflow:5000")

    experiment_name = cfg.tracking.mlflow.experiment_name
    ensure_experiment(experiment_name)

    app.state.experiment_name = experiment_name

    # DO NOT crash API if MLflow/model is not ready
    try:
        app.state.model = load_latest_from_experiment(experiment_name)
    except Exception as e:
        print(f"[WARN] model not loaded at startup: {e}")
        app.state.model = None

    yield

    logger.info("Shutting down API")


app = FastAPI(
    title="model-service",
    lifespan=lifespan,
)

# Metrics endpoint
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health(request: Request):
    model_loaded = (
        hasattr(request.app.state, "model") and request.app.state.model is not None
    )

    return {
        "status": "ok" if model_loaded else "not_ready",
        "model_loaded": model_loaded,
        "model_name": getattr(request.app.state, "model_name", None),
    }


@app.post("/predict")
@PREDICTION_LATENCY.time()
def predict(req: PredictRequest, request: Request):
    if not hasattr(request.app.state, "model") or request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not req.texts:
        raise HTTPException(status_code=400, detail="Empty input")

    PREDICTION_COUNTER.inc()

    try:
        preds = request.app.state.model.predict(pd.DataFrame({"text": req.texts}))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

    return {"predictions": preds.tolist() if hasattr(preds, "tolist") else preds}
