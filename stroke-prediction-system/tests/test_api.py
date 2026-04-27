"""Tests for the interviewer-ready stroke prediction API and preprocessing pipeline."""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from ml.preprocessing import StrokeDataPreprocessor

client = TestClient(api_main.app)


class FakePredictor:
    def __init__(self):
        self.model_type = "logistic"
        self.model_label = "Logistic Regression"
        self.model_version = "baseline-v1"
        self.threshold = 0.30

    def predict_proba(self, X):
        return [[0.22, 0.78]]

    def apply_threshold(self, probabilities):
        return (probabilities >= self.threshold).astype(int)

    def build_explanation_summary(self, X):
        return [
            {"feature": "age", "contribution": 1.2, "direction": "increases risk"},
            {"feature": "heart_disease", "contribution": 0.8, "direction": "increases risk"},
        ]


class FakePreprocessor:
    def preprocess(self, df, fit=False):
        X = pd.DataFrame(
            [
                {
                    "age": df.iloc[0]["age"],
                    "hypertension": df.iloc[0]["hypertension"],
                    "heart_disease": df.iloc[0]["heart_disease"],
                    "avg_glucose_level": df.iloc[0]["avg_glucose_level"],
                    "bmi": df.iloc[0]["bmi"] if pd.notnull(df.iloc[0]["bmi"]) else 28.1,
                    "gender_other": 1 if df.iloc[0]["gender"] == "Other" else 0,
                    "smoking_status_formerly smoked": 1 if df.iloc[0]["smoking_status"] == "formerly smoked" else 0,
                    "smoking_status_smokes": 1 if df.iloc[0]["smoking_status"] == "smokes" else 0,
                    "smoking_status_Unknown": 1 if df.iloc[0]["smoking_status"] == "Unknown" else 0,
                }
            ]
        )
        return X, None


@pytest.fixture(autouse=True)
def configure_fake_runtime(monkeypatch):
    monkeypatch.setattr(api_main, "model", FakePredictor())
    monkeypatch.setattr(api_main, "preprocessor", FakePreprocessor())
    monkeypatch.setattr(
        api_main,
        "model_metadata",
        {
            "active_model": {
                "model_type": "logistic",
                "label": "Logistic Regression",
                "version": "baseline-v1",
                "training_date": "2026-03-11T12:24:25",
                "threshold": 0.30,
                "threshold_rationale": "Lowered threshold for screening recall.",
                "why_this_version": "High recall with easier explanation.",
                "primary_objective": "High recall screening support with fewer false negatives.",
                "use_case": "Stroke risk screening support, not diagnosis.",
                "metrics": {
                    "accuracy": 0.74,
                    "precision": 0.14,
                    "recall": 0.80,
                    "f1_score": 0.23,
                    "roc_auc": 0.84,
                },
                "confusion_matrix": [[716, 256], [10, 40]],
                "explanation_method": "Signed logistic coefficients translated into plain-language factor summaries.",
            },
            "candidate_models": [
                {
                    "model_type": "logistic",
                    "label": "Logistic Regression",
                    "status": "active",
                    "selection_rationale": "High recall with interpretability.",
                    "rejection_rationale": "Not rejected.",
                    "metrics": {"accuracy": 0.74, "precision": 0.14, "recall": 0.80, "f1_score": 0.23, "roc_auc": 0.84},
                    "confusion_matrix": [[716, 256], [10, 40]],
                }
            ],
            "feature_decisions": {
                "selected_features": ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "gender_other"],
                "selected_feature_count": 6,
                "excluded_features": {"work_type": "Low predictive value."},
                "display_feature_labels": {"age": "Age"},
                "preprocessing_decisions": {"bmi_imputation": "Median fill"},
                "raw_input_fields": ["age", "gender", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "smoking_status"],
            },
            "workflow_summary": {"steps": ["Raw data intake"], "engineering_signals": ["Reusable preprocessing"]},
            "limitations": ["Portfolio system."],
            "next_steps": ["Try ensembles."],
        },
    )


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == "/model/metadata"



def test_health_endpoint_reports_loaded_runtime():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True



def test_metadata_endpoint_exposes_workflow_summary():
    response = client.get("/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["active_model"]["label"] == "Logistic Regression"
    assert "candidate_models" in data
    assert "feature_decisions" in data
    assert "workflow_summary" in data



def test_prediction_endpoint_valid_input_returns_stable_contract():
    patient_data = {
        "age": 67,
        "gender": "Male",
        "hypertension": 0,
        "heart_disease": 1,
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly smoked",
    }

    response = client.post("/predict", json=patient_data)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["threshold"] == 0.3
    assert data["active_model"] == "Logistic Regression"
    assert len(data["explanation_summary"]) >= 1



def test_prediction_endpoint_invalid_gender():
    patient_data = {
        "age": 50,
        "gender": "InvalidGender",
        "hypertension": 0,
        "heart_disease": 0,
        "avg_glucose_level": 100,
        "bmi": 25,
        "smoking_status": "never smoked",
    }

    response = client.post("/predict", json=patient_data)
    assert response.status_code == 422



def test_prediction_endpoint_missing_removed_fields_is_still_valid():
    patient_data = {
        "age": 50,
        "gender": "Female",
        "hypertension": 0,
        "heart_disease": 0,
        "avg_glucose_level": 100,
        "bmi": 25,
        "smoking_status": "never smoked",
    }

    response = client.post("/predict", json=patient_data)
    assert response.status_code == 200



def test_bmi_is_optional_and_gender_other_is_supported():
    patient_data = {
        "age": 50,
        "gender": "Other",
        "hypertension": 0,
        "heart_disease": 0,
        "avg_glucose_level": 100,
        "bmi": None,
        "smoking_status": "Unknown",
    }

    response = client.post("/predict", json=patient_data)
    assert response.status_code == 200



def test_preprocessor_uses_notebook_selected_features_only():
    df = pd.DataFrame(
        [
            {
                "age": 67,
                "gender": "Other",
                "hypertension": 1,
                "heart_disease": 0,
                "ever_married": "Yes",
                "work_type": "Private",
                "Residence_type": "Urban",
                "avg_glucose_level": 150.0,
                "bmi": 29.4,
                "smoking_status": "Unknown",
                "stroke": 1,
            }
        ]
    )
    preprocessor = StrokeDataPreprocessor()
    X, y = preprocessor.preprocess(df, fit=True)

    assert "age" in X.columns
    assert "work_type" not in X.columns
    assert "Residence_type" not in X.columns
    assert "ever_married" not in X.columns
    assert "gender_other" in X.columns
    assert X.loc[0, "gender_other"] == 1
    assert y.iloc[0] == 1


@pytest.mark.parametrize("missing_column", ["gender", "smoking_status"])
def test_preprocessor_rejects_missing_required_categorical_values(missing_column):
    df = pd.DataFrame(
        [
            {
                "age": 67,
                "gender": "Male",
                "hypertension": 1,
                "heart_disease": 0,
                "avg_glucose_level": 150.0,
                "bmi": 28.4,
                "smoking_status": "never smoked",
                "stroke": 1,
            }
        ]
    )
    df.loc[0, missing_column] = pd.NA

    preprocessor = StrokeDataPreprocessor()

    with pytest.raises(ValueError, match=f"Column '{missing_column}' contains missing values"):
        preprocessor.preprocess(df, fit=True)


def test_preprocessor_rejects_training_data_when_all_bmi_values_are_missing():
    df = pd.DataFrame(
        [
            {
                "age": 67,
                "gender": "Male",
                "hypertension": 1,
                "heart_disease": 0,
                "avg_glucose_level": 150.0,
                "bmi": None,
                "smoking_status": "never smoked",
                "stroke": 1,
            }
        ]
    )

    preprocessor = StrokeDataPreprocessor()

    with pytest.raises(ValueError, match="BMI median cannot be learned"):
        preprocessor.preprocess(df, fit=True)
