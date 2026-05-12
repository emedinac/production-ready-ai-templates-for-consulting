class InferenceWrapper(mlflow.pyfunc.PythonModel):
    """Unified inference wrapper supporting both sklearn and transformer models.

    Enforces single artifact contract:
    - model: The trained model (sklearn or HF transformer)
    - tokenizer: Optional tokenizer for transformer models
    - preprocessor: Optional feature preprocessor for sklearn models
    - type: Model type identifier ("transformer" or "sklearn")
    """

    def load_context(self, context):
        import joblib

        self.bundle = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input):
        texts = model_input["text"]

        model = self.bundle["model"]
        tokenizer = self.bundle.get("tokenizer")
        preprocessor = self.bundle.get("preprocessor")
        model_type = self.bundle.get("type", "sklearn")

        # Transformer models use tokenizer
        if model_type == "transformer" and tokenizer:
            features = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )
            return model(**features).logits

        # Sklearn models use preprocessor if available
        if preprocessor:
            features = preprocessor.transform(texts)
        else:
            features = texts

        return model.predict(features)
