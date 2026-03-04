"""
FastAPI application for stroke prediction.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

from api.schemas import (
    PatientInput, PredictionResponse, PredictionWithExplanation,
    HealthCheck, ModelInfo
)
from ml.preprocessing import StrokeDataPreprocessor
from ml.train import StrokePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stroke Prediction API",
    description="Production-ready API for stroke risk prediction with ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / 'models'

# Global variables for model and preprocessor
model = None
preprocessor = None

def load_model_and_preprocessor():
    """Load trained model and preprocessor."""
    global model, preprocessor
    
    try:
        # Load preprocessor
        preprocessor = StrokeDataPreprocessor()
        preprocessor.load(MODELS_DIR / 'preprocessor.pkl')
        logger.info("✅ Preprocessor loaded successfully")
        
        # Load model
        model = StrokePredictor()
        model.load(MODELS_DIR / 'stroke_model_production.pkl')
        logger.info(f"✅ Model loaded successfully: {model.model_type}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False


def get_risk_level(probability: float) -> str:
    """Determine risk level based on probability."""
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting Stroke Prediction API...")
    success = load_model_and_preprocessor()
    if not success:
        logger.warning("⚠️  Model not loaded - some endpoints may not work")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Stroke Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck, tags=["General"])
async def health_check():
    """Health check endpoint."""
    model_loaded = model is not None and preprocessor is not None
    
    return HealthCheck(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_type=model.model_type,
        training_date=model.training_metrics.get('training_date'),
        metrics={
            'accuracy': model.training_metrics.get('accuracy'),
            'precision': model.training_metrics.get('precision'),
            'recall': model.training_metrics.get('recall'),
            'f1_score': model.training_metrics.get('f1_score'),
            'roc_auc': model.training_metrics.get('roc_auc')
        },
        feature_importance=model.feature_importance
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_stroke(patient: PatientInput):
    """
    Predict stroke risk for a patient.
    
    Returns prediction with probability and risk level.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([patient.dict()])
        
        # Preprocess
        X, _ = preprocessor.preprocess(input_data, fit=False)
        
        # Predict
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        probability = float(probabilities[1])  # Probability of stroke
        
        # Determine risk level
        risk_level = get_risk_level(probability)
        
        # Confidence is the max probability
        confidence = float(max(probabilities))
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/explain", response_model=PredictionWithExplanation, tags=["Predictions"])
async def predict_with_explanation(patient: PatientInput):
    """
    Predict stroke risk with feature-contribution explanations.
    
    Returns prediction along with approximate feature contributions.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([patient.dict()])
        
        # Preprocess
        X, _ = preprocessor.preprocess(input_data, fit=False)
        
        # Predict
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        probability = float(probabilities[1])
        
        risk_level = get_risk_level(probability)
        confidence = float(max(probabilities))
        
        # Get feature contributions (using feature importance as proxy)
        # In production, this would use SHAP
        feature_contributions = {}
        if model.feature_importance:
            for feature, importance in model.feature_importance.items():
                try:
                    feature_value = float(X[feature].values[0])
                    contribution = importance * feature_value * probability
                    feature_contributions[feature] = round(contribution, 4)
                except:
                    pass
        
        # Get top 5 risk factors
        top_factors = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        top_risk_factors = [
            {"feature": feat, "contribution": float(contrib)}
            for feat, contrib in top_factors
        ]
        
        return PredictionWithExplanation(
            prediction=int(prediction),
            probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            feature_contributions=feature_contributions,
            top_risk_factors=top_risk_factors,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction with explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
