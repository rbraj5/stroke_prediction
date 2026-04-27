"""Pydantic schemas for stable API contracts used by the dashboard and tests."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatientInput(BaseModel):
    """Prediction input schema for the current stable dashboard/API contract."""

    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    gender: str = Field(..., description="Gender: Male, Female, or Other")
    hypertension: int = Field(..., ge=0, le=1, description="0: No, 1: Yes")
    heart_disease: int = Field(..., ge=0, le=1, description="0: No, 1: Yes")
    avg_glucose_level: float = Field(..., ge=0, le=300, description="Average glucose level (mg/dL)")
    bmi: Optional[float] = Field(None, ge=10, le=100, description="Body Mass Index")
    smoking_status: str = Field(..., description="Smoking status: never smoked, formerly smoked, smokes, or Unknown")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value: str) -> str:
        allowed = ["male", "female", "other", "Male", "Female", "Other"]
        if value not in allowed:
            raise ValueError(f"Gender must be one of: {allowed}")
        return value

    @field_validator("smoking_status")
    @classmethod
    def validate_smoking(cls, value: str) -> str:
        allowed = ["never smoked", "formerly smoked", "smokes", "Unknown"]
        if value not in allowed:
            raise ValueError(f"smoking_status must be one of: {allowed}")
        return value

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 67,
                "gender": "Male",
                "hypertension": 0,
                "heart_disease": 1,
                "avg_glucose_level": 228.69,
                "bmi": 36.6,
                "smoking_status": "formerly smoked",
            }
        }
    )


class ExplanationFactor(BaseModel):
    """Stable explanation item so future models can swap methods without changing the response shape."""

    feature: str
    contribution: float
    direction: str


class PredictionResponse(BaseModel):
    """Prediction response used by both the dashboard and API tests."""

    prediction: int = Field(..., description="0: lower screening risk, 1: elevated screening risk")
    probability: float = Field(..., description="Predicted probability of stroke risk (0-1)")
    threshold: float = Field(..., description="Decision threshold used for classification")
    risk_level: str = Field(..., description="Low, Moderate, or High screening risk")
    confidence: float = Field(..., description="Model confidence (0-1)")
    active_model: str = Field(..., description="Currently active model label")
    model_version: str = Field(..., description="Version of the active model artifact")
    explanation_summary: List[ExplanationFactor] = Field(default_factory=list)
    note: str = Field(..., description="Plain-language reminder about the intended use of the model")
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthCheck(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool
    version: str = "2.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(protected_namespaces=())


class ModelMetadataResponse(BaseModel):
    """Metadata response that helps the UI tell the full workflow story."""

    active_model: Dict[str, object]
    candidate_models: List[Dict[str, object]]
    feature_decisions: Dict[str, object]
    workflow_summary: Dict[str, object]
    limitations: List[str]
    next_steps: List[str]

    model_config = ConfigDict(protected_namespaces=())
