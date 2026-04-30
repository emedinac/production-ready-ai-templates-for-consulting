from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from serving.loader import load_production_model
from serving.schema import PredictRequest
from configs.validate import load_and_validate_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_and_validate_config()
    app.state.model_name = cfg.tracking.mlflow.experiment_name
    app.state.model = load_production_model(app.state.model_name)
    yield


app = FastAPI(title="{{ project_name }} model service", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_name": getattr(request.app.state, "model_name", None),
        "model_loaded": hasattr(request.app.state, "model"),
    }


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    preds = model.predict(req.texts)
    if hasattr(preds, "tolist"):
        preds = preds.tolist()

    return {"predictions": preds}
