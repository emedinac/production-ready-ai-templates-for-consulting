from fastapi import FastAPI
from serving.loader import load_production_model
from serving.schema import PredictRequest
from configs.validate import load_and_validate_config


app = FastAPI()

cfg = load_and_validate_config()
model = load_production_model(cfg.tracking.mlflow.experiment_name)


@app.post("/predict")
def predict(req: PredictRequest):
    preds = model.predict(req.texts)
    return {"predictions": preds.tolist()}
