from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(StrictConfig):
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    tags: list[str] = Field(default_factory=list)


class TaskConfig(StrictConfig):
    type: Literal["text_classification", "tabular_regression"]
    problem_type: Literal["binary", "multiclass", "regression"]


class HuggingFaceSource(StrictConfig):
    name: str = Field(min_length=1)
    version: str | None = None


class LocalSource(StrictConfig):
    path: str | None = None


class S3Source(StrictConfig):
    bucket: str | None = None
    key: str | None = None


class DataSource(StrictConfig):
    type: Literal["huggingface", "local", "s3"]
    huggingface: HuggingFaceSource | None = None
    local: LocalSource | None = None
    s3: S3Source | None = None

    @model_validator(mode="after")
    def validate_source(self):
        if self.type == "huggingface" and not self.huggingface:
            raise ValueError("huggingface config required")
        if self.type == "local" and (not self.local or not self.local.path):
            raise ValueError("local.path required")
        if self.type == "s3" and (not self.s3 or not self.s3.bucket or not self.s3.key):
            raise ValueError("s3.bucket and s3.key required")
        return self


class SplitConfig(StrictConfig):
    train: float = Field(gt=0, lt=1)
    validation: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)

    @field_validator("test")
    @classmethod
    def check_sum(cls, v: float, info):
        total = v + info.data.get("train", 0) + info.data.get("validation", 0)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("splits must sum to 1.0")
        return v


class TextPreprocessingConfig(StrictConfig):
    lowercase: bool
    max_length: int = Field(ge=1)


class TabularPreprocessingConfig(StrictConfig):
    normalize: bool
    handle_missing: Literal["mean", "median", "mode", "drop"]


class PreprocessingConfig(StrictConfig):
    text: TextPreprocessingConfig | None = None
    tabular: TabularPreprocessingConfig | None = None


class DataConfig(StrictConfig):
    source: DataSource
    split: SplitConfig
    preprocessing: PreprocessingConfig | None = None
    processed_dir: str = "{{project_name}}/data/processed"


class TransformerConfig(StrictConfig):
    name: str = Field(min_length=1)
    dropout: float = Field(ge=0, lt=1)


class XGBoostConfig(StrictConfig):
    max_depth: int = Field(ge=1)
    n_estimators: int = Field(ge=1)


class LinearConfig(StrictConfig):
    fit_intercept: bool


class ModelConfig(StrictConfig):
    type: Literal["transformer", "xgboost", "linear"]
    transformer: TransformerConfig | None = None
    xgboost: XGBoostConfig | None = None
    linear: LinearConfig | None = None

    @model_validator(mode="after")
    def validate_model(self):
        if self.type == "transformer" and not self.transformer:
            raise ValueError("transformer config required")
        if self.type == "xgboost" and not self.xgboost:
            raise ValueError("xgboost config required")
        if self.type == "linear" and not self.linear:
            raise ValueError("linear config required")
        return self


class SchedulerConfig(StrictConfig):
    type: Literal["linear", "none"]
    warmup_steps: int | None = None

    @model_validator(mode="after")
    def validate_scheduler(self):
        if self.type == "linear" and self.warmup_steps is None:
            raise ValueError("warmup_steps required for linear scheduler")
        return self


class TrainingConfig(StrictConfig):
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    optimizer: str = Field(min_length=1)
    scheduler: SchedulerConfig


class ValidationConfig(StrictConfig):
    type: Literal["holdout", "kfold"]
    folds: int | None = None

    @model_validator(mode="after")
    def validate_folds(self):
        if self.type == "kfold" and (self.folds is None or self.folds < 2):
            raise ValueError("folds must be at least 2 for kfold validation")
        return self


class EvaluationConfig(StrictConfig):
    metrics: list[str] = Field(min_length=1)
    validation: ValidationConfig


class RuntimeConfig(StrictConfig):
    device: Literal["auto", "cpu", "cuda"]
    num_workers: int = Field(ge=0)


class MLflowConfig(StrictConfig):
    enabled: bool
    experiment_name: str = Field(min_length=1)


class TrackingConfig(StrictConfig):
    mlflow: MLflowConfig


class ArtifactsConfig(StrictConfig):
    metrics_dir: str = "metrics"


class OutputConfig(StrictConfig):
    model_path: Path
    metrics_path: Path
    logs_path: Path


class FullConfig(StrictConfig):
    experiment: ExperimentConfig
    task: TaskConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig
    tracking: TrackingConfig
    artifacts: ArtifactsConfig
    output: OutputConfig

    @model_validator(mode="after")
    def validate_cross_logic(self):
        # Example: prevent invalid combos
        if self.task.type == "text_classification" and self.model.type == "xgboost":
            raise ValueError("xgboost not supported for text_classification")
        return self
