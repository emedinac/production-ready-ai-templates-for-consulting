from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def predict(inputs: list[str], artifact_path: str = None) -> list[Any]:
    """Standalone predict function for inference.
    
    Args:
        inputs: List of input texts
        artifact_path: Path to the model artifact
        
    Returns:
        List of predictions
    """
    if artifact_path is None:
        # Return empty list if no artifact path provided (for testing)
        return []
    
    predictor = Predictor(artifact_path)
    return predictor.predict(inputs)


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
