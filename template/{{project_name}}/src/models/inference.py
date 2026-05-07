from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class Predictor:
    def __init__(self, artifact_path: str):
        artifact = joblib.load(Path(artifact_path))

        self.model = artifact["model"]
        self.preprocessor = artifact.get("preprocessor")

    def predict(self, inputs: list[str]) -> list[Any]:
        df = pd.DataFrame({"text": inputs})

        if self.preprocessor:
            features = self.preprocessor.transform(df["text"])
        else:
            features = df

        predictions = self.model.predict(features)

        return predictions.tolist()
