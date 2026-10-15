"""Pydantic request/response schemas (validation + Swagger docs)."""
from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class GeographyEnum(str, Enum):
    france = "France"
    spain = "Spain"
    germany = "Germany"


class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"


class Customer(BaseModel):
    """Features of a single customer. Value ranges are validated automatically."""

    CreditScore: int = Field(..., ge=300, le=900, examples=[650])
    Geography: GeographyEnum = Field(..., examples=["France"])
    Gender: GenderEnum = Field(..., examples=["Female"])
    Age: int = Field(..., ge=18, le=100, examples=[42])
    Tenure: int = Field(..., ge=0, le=10, examples=[3])
    Balance: float = Field(..., ge=0, examples=[75000.0])
    NumOfProducts: int = Field(..., ge=1, le=4, examples=[2])
    HasCrCard: int = Field(..., ge=0, le=1, examples=[1])
    IsActiveMember: int = Field(..., ge=0, le=1, examples=[1])
    EstimatedSalary: float = Field(..., ge=0, examples=[100000.0])

    # Unknown fields are rejected (422) instead of being silently ignored.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "CreditScore": 650, "Geography": "France", "Gender": "Female",
                "Age": 42, "Tenure": 3, "Balance": 75000.0, "NumOfProducts": 2,
                "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 100000.0,
            }
        },
    )


class BatchRequest(BaseModel):
    customers: List[Customer] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Churn probability in [0, 1]")
    churn: bool = Field(..., description="Predicted class at threshold 0.5")
    risk_level: str = Field(..., description="low / medium / high")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_roc_auc: float | None = None
