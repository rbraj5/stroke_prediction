"""
Pydantic schemas for API request and response validation.
"""
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatientInput(BaseModel):
    """Input schema for patient data."""

    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    gender: str = Field(..., description="Gender: Male, Female, or Other")
    hypertension: int = Field(..., ge=0, le=1, description="0: No, 1: Yes")
    heart_disease: int = Field(..., ge=0, le=1, description="0: No, 1: Yes")
    ever_married: str = Field(..., description="Ever married: Yes or No")
    work_type: str = Field(..., description="Type of work: Private, Self-employed, Govt_job, children, or Never_worked")
    Residence_type: str = Field(..., description="Residence: Urban or Rural")
    avg_glucose_level: float = Field(..., ge=0, le=300, description="Average glucose level (mg/dL)")
    bmi: Optional[float] = Field(None, ge=10, le=100, description="Body Mass Index")
    smoking_status: str = Field(..., description="Smoking status: never smoked, formerly smoked, smokes, or Unknown")

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v: str) -> str:
        allowed = ['male', 'female', 'other', 'Male', 'Female', 'Other']
        if v not in allowed:
            raise ValueError(f'Gender must be one of: {allowed}')
        return v

    @field_validator('ever_married')
    @classmethod
    def validate_married(cls, v: str) -> str:
        allowed = ['yes', 'no', 'Yes', 'No']
        if v not in allowed:
            raise ValueError(f'ever_married must be one of: {allowed}')
        return v

    @field_validator('work_type')
    @classmethod
    def validate_work(cls, v: str) -> str:
        allowed = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
        if v not in allowed:
            raise ValueError(f'work_type must be one of: {allowed}')
        return v

    @field_validator('Residence_type')
    @classmethod
    def validate_residence(cls, v: str) -> str:
        allowed = ['Urban', 'Rural', 'urban', 'rural']
        if v not in allowed:
            raise ValueError(f'Residence_type must be one of: {allowed}')
        return v

    @field_validator('smoking_status')
    @classmethod
    def validate_smoking(cls, v: str) -> str:
        allowed = ['never smoked', 'formerly smoked', 'smokes', 'Unknown']
        if v not in allowed:
            raise ValueError(f'smoking_status must be one of: {allowed}')
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 67,
                "gender": "Male",
                "hypertension": 0,
                "heart_disease": 1,
                "ever_married": "Yes",
                "work_type": "Private",
                "Residence_type": "Urban",
                "avg_glucose_level": 228.69,
                "bmi": 36.6,
                "smoking_status": "formerly smoked"
            }
        }
    )


class PredictionResponse(BaseModel):
    """Response schema for predictions."""

    prediction: int = Field(..., description="0: No stroke risk, 1: Stroke risk")
    probability: float = Field(..., description="Probability of stroke (0-1)")
    risk_level: str = Field(..., description="Low, Medium, or High risk")
    confidence: float = Field(..., description="Model confidence (0-1)")
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 1,
                "probability": 0.78,
                "risk_level": "High",
                "confidence": 0.78,
                "timestamp": "2024-02-08T10:30:00"
            }
        }
    )


class PredictionWithExplanation(PredictionResponse):
    """Extended response with SHAP explanations."""

    feature_contributions: Dict[str, float] = Field(
        ...,
        description="Approximate feature contributions"
    )
    top_risk_factors: List[Dict[str, float]] = Field(
        ...,
        description="Top 5 features contributing to the prediction"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 1,
                "probability": 0.78,
                "risk_level": "High",
                "confidence": 0.78,
                "timestamp": "2024-02-08T10:30:00",
                "feature_contributions": {
                    "age": 0.25,
                    "heart_disease": 0.18,
                    "avg_glucose_level": 0.15
                },
                "top_risk_factors": [
                    {"feature": "age", "contribution": 0.25},
                    {"feature": "heart_disease", "contribution": 0.18}
                ]
            }
        }
    )


class HealthCheck(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(protected_namespaces=())


class ModelInfo(BaseModel):
    """Model information response."""

    model_type: str
    training_date: Optional[str]
    metrics: Dict
    feature_importance: Optional[Dict[str, float]]

    model_config = ConfigDict(protected_namespaces=())
