from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app

from serving.loader import load_production_model
from serving.schema import PredictRequest
from configs.validate import load_and_validate_config
import pandas as pd

# PROMETHEUS METRICS
PREDICTION_COUNTER = Counter(
    "inference_requests_total",
    "Total number of inference requests",
)

PREDICTION_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_and_validate_config()

    app.state.model_name = cfg.tracking.mlflow.experiment_name
    app.state.model = load_production_model(app.state.model_name)

    yield


app = FastAPI(
    title="CHANGE_ME model service",
    lifespan=lifespan,
)

# PROMETHEUS /metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_name": getattr(request.app.state, "model_name", None),
        "model_loaded": hasattr(request.app.state, "model"),
    }


@app.post("/predict")
@PREDICTION_LATENCY.time()
def predict(req: PredictRequest, request: Request):
    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(503, "Model not loaded")

    if not req.texts:
        raise HTTPException(400, "Empty input")

    PREDICTION_COUNTER.inc()

    try:
        preds = model.predict(pd.DataFrame({"text": req.texts}))
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {str(e)}")

    return {"predictions": preds.tolist() if hasattr(preds, "tolist") else preds}
