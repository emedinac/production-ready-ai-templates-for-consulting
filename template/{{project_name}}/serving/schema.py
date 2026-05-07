from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)
