from pydantic import BaseModel, field_validator, model_validator
from typing import Literal, Optional, List


class ExperimentConfig(BaseModel):
    name: str
    seed: int
    tags: List[str]


class TaskConfig(BaseModel):
    type: Literal["text_classification", "tabular_regression"]
    problem_type: Literal["binary", "multiclass", "regression"]


class HuggingFaceSource(BaseModel):
    name: str
    version: Optional[str]


class LocalSource(BaseModel):
    path: Optional[str]


class S3Source(BaseModel):
    bucket: Optional[str]
    key: Optional[str]


class DataSource(BaseModel):
    type: Literal["huggingface", "local", "s3"]
    huggingface: Optional[HuggingFaceSource]
    local: Optional[LocalSource]
    s3: Optional[S3Source]

    @model_validator(mode="after")
    def validate_source(self):
        if self.type == "huggingface" and not self.huggingface:
            raise ValueError("huggingface config required")
        if self.type == "local" and (not self.local or not self.local.path):
            raise ValueError("local.path required")
        if self.type == "s3" and (not self.s3 or not self.s3.bucket or not self.s3.key):
            raise ValueError("s3.bucket and s3.key required")
        return self


class SplitConfig(BaseModel):
    train: float
    validation: float
    test: float

    @field_validator("test")
    def check_sum(cls, v, info):
        total = v + info.data["train"] + info.data["validation"]
        if abs(total - 1.0) > 1e-6:
            raise ValueError("splits must sum to 1.0")
        return v


class DataConfig(BaseModel):
    source: DataSource
    split: SplitConfig


class TransformerConfig(BaseModel):
    name: str
    dropout: float


class XGBoostConfig(BaseModel):
    max_depth: int
    n_estimators: int


class LinearConfig(BaseModel):
    fit_intercept: bool


class ModelConfig(BaseModel):
    type: Literal["transformer", "xgboost", "linear"]
    transformer: Optional[TransformerConfig]
    xgboost: Optional[XGBoostConfig]
    linear: Optional[LinearConfig]

    @model_validator(mode="after")
    def validate_model(self):
        if self.type == "transformer" and not self.transformer:
            raise ValueError("transformer config required")
        if self.type == "xgboost" and not self.xgboost:
            raise ValueError("xgboost config required")
        if self.type == "linear" and not self.linear:
            raise ValueError("linear config required")
        return self


class SchedulerConfig(BaseModel):
    type: Literal["linear", "none"]
    warmup_steps: Optional[int]


class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    scheduler: SchedulerConfig


class ValidationConfig(BaseModel):
    type: Literal["holdout", "kfold"]
    folds: Optional[int]


class EvaluationConfig(BaseModel):
    metrics: List[str]
    validation: ValidationConfig


class RuntimeConfig(BaseModel):
    device: Literal["auto", "cpu", "cuda"]
    num_workers: int


class MLflowConfig(BaseModel):
    enabled: bool
    experiment_name: str


class TrackingConfig(BaseModel):
    mlflow: MLflowConfig


class OutputConfig(BaseModel):
    model_path: str
    metrics_path: str
    logs_path: str


class FullConfig(BaseModel):
    experiment: ExperimentConfig
    task: TaskConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig
    tracking: TrackingConfig
    output: OutputConfig

    @model_validator(mode="after")
    def validate_cross_logic(self):
        # Example: prevent invalid combos
        if self.task.type == "text_classification" and self.model.type == "xgboost":
            raise ValueError("xgboost not supported for text_classification")
        return self
