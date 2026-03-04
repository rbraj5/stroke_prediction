"""
Tests for the Stroke Prediction API.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns correct info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_prediction_endpoint_valid_input():
    """Test prediction with valid patient data."""
    patient_data = {
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
    
    response = client.post("/predict", json=patient_data)
    
    # 503 is expected when model artifacts are not available.
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "risk_level" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1
        assert data["risk_level"] in ["Low", "Medium", "High"]


def test_prediction_endpoint_invalid_age():
    """Test prediction rejects invalid age."""
    patient_data = {
        "age": 150,  # Invalid age
        "gender": "Male",
        "hypertension": 0,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 100,
        "bmi": 25,
        "smoking_status": "never smoked"
    }
    
    response = client.post("/predict", json=patient_data)
    assert response.status_code == 422  # Validation error


def test_prediction_endpoint_invalid_gender():
    """Test prediction rejects invalid gender."""
    patient_data = {
        "age": 50,
        "gender": "InvalidGender",
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 100,
        "bmi": 25,
        "smoking_status": "never smoked"
    }
    
    response = client.post("/predict", json=patient_data)
    assert response.status_code == 422


def test_prediction_endpoint_missing_field():
    """Test prediction rejects missing required fields."""
    patient_data = {
        "age": 50,
        "gender": "Male",
        # Missing hypertension
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 100,
        "bmi": 25,
        "smoking_status": "never smoked"
    }
    
    response = client.post("/predict", json=patient_data)
    assert response.status_code == 422


def test_explain_endpoint():
    """Test prediction with explanation endpoint."""
    patient_data = {
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
    
    response = client.post("/predict/explain", json=patient_data)
    
    # 503 is expected when model artifacts are not available.
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "feature_contributions" in data
        assert "top_risk_factors" in data


def test_optional_bmi():
    """Test that BMI is optional (will be filled with median)."""
    patient_data = {
        "age": 50,
        "gender": "Female",
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 100,
        "bmi": None,  # Optional
        "smoking_status": "never smoked"
    }
    
    response = client.post("/predict", json=patient_data)
    # Should work (bmi will be filled with median in preprocessing)
    assert response.status_code in [200, 503]  # 503 if model not loaded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
