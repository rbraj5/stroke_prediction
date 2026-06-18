"""
FastAPI inference layer for the stroke prediction portfolio project.

This API keeps the public contract stable even if the active model changes.
The dashboard reads the same metadata file that training writes, so the UI can
show honest workflow details instead of hardcoded claims.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import HealthCheck, ModelMetadataResponse, PatientInput, PredictionResponse
from ml.preprocessing import StrokeDataPreprocessor
from ml.train import StrokePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stroke Prediction API",
    description="Portfolio-ready API for stroke risk screening predictions and workflow metadata.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

model: StrokePredictor | None = None
preprocessor: StrokeDataPreprocessor | None = None
model_metadata: Dict[str, Any] | None = None


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_model_and_preprocessor() -> bool:
    """Load the active model, preprocessor, and metadata bundle for inference."""
    global model, preprocessor, model_metadata

    try:
        preprocessor = StrokeDataPreprocessor()
        preprocessor.load(MODELS_DIR / "preprocessor.pkl")
        logger.info("Preprocessor loaded successfully")

        model = StrokePredictor()
        model.load(MODELS_DIR / "stroke_model_production.pkl")
        logger.info("Model loaded successfully: %s", model.model_type)

        model_metadata = load_json(MODELS_DIR / "model_metadata.json")
        logger.info("Model metadata loaded successfully")
        return True
    except Exception as exc:
        logger.error("Error loading model artifacts: %s", exc)
        return False


def get_risk_level(probability: float) -> str:
    """Convert a raw probability into a plain-language screening risk band."""
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Moderate"
    return "High"


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Starting Stroke Prediction API...")
    success = load_model_and_preprocessor()
    if not success:
        logger.warning("Model artifacts were not loaded; prediction endpoints will be unavailable.")


@app.get("/", tags=["General"])
async def root() -> Dict[str, str]:
    return {
        "message": "Stroke Prediction Portfolio API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "metadata": "/model/metadata",
    }


@app.get("/health", response_model=HealthCheck, tags=["General"])
async def health_check() -> HealthCheck:
    loaded = model is not None and preprocessor is not None and model_metadata is not None
    return HealthCheck(status="healthy" if loaded else "degraded", model_loaded=loaded)


@app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Model"])
async def get_model_metadata() -> ModelMetadataResponse:
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded")
    return ModelMetadataResponse(**model_metadata)


@app.get("/model/info", response_model=ModelMetadataResponse, tags=["Model"])
async def get_model_info() -> ModelMetadataResponse:
    """Alias kept so older links still work while the dashboard migrates to /model/metadata."""
    return await get_model_metadata()


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_stroke(patient: PatientInput) -> PredictionResponse:
    if model is None or preprocessor is None or model_metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_data = pd.DataFrame([patient.model_dump()])
        X, _ = preprocessor.preprocess(input_data, fit=False)

        probabilities = model.predict_proba(X)[0]
        probability = float(probabilities[1])
        prediction = int(model.apply_threshold(pd.Series([probability]).to_numpy())[0])
        confidence = float(max(probabilities))
        explanation_summary = model.build_explanation_summary(X)

        active_model = model_metadata["active_model"]
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            threshold=float(active_model["threshold"]),
            risk_level=get_risk_level(probability),
            confidence=confidence,
            active_model=str(active_model["label"]),
            model_version=str(active_model["version"]),
            explanation_summary=explanation_summary,
            note="This is a screening-support prediction for portfolio demonstration, not a clinical diagnosis.",
            timestamp=datetime.now(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
